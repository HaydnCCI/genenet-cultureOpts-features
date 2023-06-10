# Local imports
import os
import datetime, time
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Third party imports
import uvicorn
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from io import StringIO
import pandas as pd

from ms.functions import get_model_response

## For data transfer
import json

# Modules
from modules.gene_modelling.data_loading import (get_gene_modelling_data,
                                                  data_preprocessing
                                                )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data/uploads")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "data/downloads")


# Paths
## Step I/O folders
project_root_folder = f"./data/gene_modelling"
step_1_root_data_folder = f"{project_root_folder}/step_0"
step_1_output_folder = f"{project_root_folder}/step_1"
step_2_output_folder = f"{project_root_folder}/step_2"
step_3_output_folder = f"{project_root_folder}/step_3"

## Step 1
### Input
step_1_metadata_output = f"{step_1_output_folder}/metadata.csv"
step_1_counts_output = f"{step_1_output_folder}/preprocessed_counts.csv"
### Output
step_1_nr_output = f"{step_1_output_folder}/modelling_nr.csv"
step_1_tpm_output = f"{step_1_output_folder}/modelling_tpm.csv"

## Step 2
### Inputs
step_2_nr_input_path = f"{step_1_output_folder}/modelling_nr.csv"
step_2_tpm_input_path = f"{step_1_output_folder}/modelling_tpm.csv"
### Output
step_2_delta_shapes_nr_path = f"{step_2_output_folder}/delta_shapes_nr.csv"
step_2_delta_shapes_tpm_path = f"{step_2_output_folder}/delta_shapes_nr.csv"
step_2_nr_features_path = f"{step_2_output_folder}/nr_features.csv" 
step_2_tpm_features_path = f"{step_2_output_folder}/tpm_features.csv" 
step_2_filtered_nr_path = f"{step_2_output_folder}/filtered_nr.csv"
step_2_filtered_tpm_path = f"{step_2_output_folder}/filtered_tpm.csv"

## Step 3
### Input
step_3_input_metadata_path = f'{step_1_output_folder}/metadata.csv'
step_3_input_nr_modelling_df_path = f"{step_2_output_folder}/filtered_nr.csv"
step_3_input_nr_features_path = f"{step_2_output_folder}/nr_features.csv"
step_3_input_modelling_df_path = f"{step_2_output_folder}/filtered_tpm.csv"
step_3_input_features_path = f"{step_2_output_folder}/tpm_features.csv"
### Output
#### Baseline results
step_3_random_gene_set_path = f"{step_2_output_folder}/random_gene_set.csv"
step_3_top_corr_gene_set_path = f"{step_2_output_folder}/top_corr_gene_set.csv"
#### DESeq2 results
step_3_deseq2_compared_pairs_path = f"{step_3_output_folder}/deseq2_compared_pairs.csv"
step_3_deseq2_res_path = f"{step_3_output_folder}/deseq2_results.csv"
step_3_deseq2_deg_res_path = f"{step_3_output_folder}/deseq2_degs_results.csv"
step_3_deseq2_deg_counts_path = f"{step_3_output_folder}/deg_counts.csv"
step_3_deseq2_percentile_counts_path = f"{step_3_output_folder}/deseq2_percentile_counts.csv"
#### Reactome results
step_3_reactome_folder_path = f"{step_3_output_folder}/reactome"
step_3_reactome_report_folder_path = f"{step_3_output_folder}/reactome-report"
step_3_reactome_firework_folder_path = f"{step_3_reactome_report_folder_path}/fireworks"
step_3_reactome_link_path = f"{step_3_reactome_report_folder_path}/reactome_links.csv"
step_3_reactome_report_path = f"{step_3_reactome_report_folder_path}/report.pdf"
step_3_reactome_sig_genes_path = f"{step_3_reactome_report_folder_path}/reactome_genes.csv"
#### Literature results
step_3_lit_genes_path = f"{step_3_output_folder}/literature_genes.csv"
step_3_filtered_modelling_df_path = f"{step_3_output_folder}/filtered_modelling_df.csv"



## About the control condition
control_col = 'drug'
control_val = "untreated"
vehicle_control_val = "dmso"

## Column names definitions
### About the meta column names
sample_col = 'sample'
raw_gene_col = 'Name'
gene_col = 'gene_id'
time_col = "hours"
## Specifying cols
select_meta_groupby_cols = ['drug', 'hours', 'sample', 'duplicate_id']
### Response column name
target_response_col = "beat_per_min"

# Structuring cols
meta_cols = [control_col, sample_col]
new_meta_col = [f"{control_col}_{sample_col}"]

## For time-series mode
next_time_point_prediction = True
keep_next_tp = [48, 168]



# Initialize FastAPI app
app = FastAPI()

# Input for data validation
class Input(BaseModel):
    concavity_mean: float = Field(..., gt=0)
    concave_points_mean: float = Field(..., gt=0)
    perimeter_se: float = Field(..., gt=0)
    area_se: float = Field(..., gt=0)
    texture_worst: float = Field(..., gt=0)
    area_worst: float = Field(..., gt=0)

    class Config:
        schema_extra = {
            "concavity_mean": 0.3001,
            "concave_points_mean": 0.1471,
            "perimeter_se": 8.589,
            "area_se": 153.4,
            "texture_worst": 17.33,
            "area_worst": 2019.0,
        }


# Ouput for data validation
class Output(BaseModel):
    label: str
    prediction: int

@app.get("/", tags=["Genenet's Product - Bioreactor Culture Optimization APIs"], status_code=200)
def root() -> dict:
    # """Root Get"""
    # return {"msg": "Hello, World!"}
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the Genenet's Product - Bioreactor Culture Optimization APIs</h1>"
    "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

@app.put('/upload_metadata', tags=["Data Loading"])
async def upload_metadata(file: UploadFile):
    """Upload metadata for the gene expression data"""
    if file.content_type != "text/csv":
        raise HTTPException(400, detail="Invalid document type")
    else:
        contents = file.file.read()
        file_ext = os.path.splitext(file.filename)[1]
#         new_filename = f"{os.path.splitext(file.filename)[0]}{file_ext}"
        new_filename = f"metadata.csv"
        SAVE_FILE_PATH = os.path.join(UPLOAD_DIR, new_filename)

        s = str(contents,'utf-8')
        data = StringIO(s)
        df = pd.read_csv(data)
        df.to_csv(SAVE_FILE_PATH)
        data.close()
        file.file.close()

        return {"Uploaded filename": file.filename,
                "Saved filename": new_filename
                }

@app.put('/upload_bioreactor_measurements', tags=["Data Loading"])
async def upload_bioreactor_measurements(file: UploadFile):
    """Upload metadata for the gene info data"""
    if file.content_type != "text/csv":
        raise HTTPException(400, detail="Invalid document type")
    else:
        contents = file.file.read()
        file_ext = os.path.splitext(file.filename)[1]
#         new_filename = f"{os.path.splitext(file.filename)[0]}{file_ext}"
        new_filename = f"measurements.csv"
        SAVE_FILE_PATH = os.path.join(UPLOAD_DIR, new_filename)

        s = str(contents,'utf-8')
        data = StringIO(s)
        df = pd.read_csv(data)
        df.to_csv(SAVE_FILE_PATH)
        data.close()
        file.file.close()

        return {"Uploaded filename": file.filename,
                "Saved filename": new_filename
                }


@app.get('/load_uploaded_data', tags=["Data Loading"])
async def load_uploaded_gene_data(target_response_col="beat_per_min"):
    """Transform the uploaded gene expression and metadata into a modelling dataframe.
    """
    try:
        # TODO

        return {
            "Modelling dataframe has been created."
            }
    except:
        return {
            "Failed to create the required modelling dataframe."
        }
    
@app.get('/split_test_dataset', tags=["Gene-based AI modelling"])
def split_test_dataset():
    pass
    return {
        "ok"
    }

################################################################################
# Classical methods
################################################################################

@app.post('/run_classical_features_selection', tags=["Classical method"])
async def run_classical_features_selection():
    """Return ..."""
    return {
        "ok"
    }

@app.post('/run_classical_DoE', tags=["Classical method"])
async def run_classical_DoE():
    """Return ..."""
    return {
        "ok"
    }


################################################################################
# AI / ML modelling
################################################################################


@app.post('/run_AI_features_selection', tags=["AI method"])
async def run_AI_features_selection():
    """Return ..."""
    return {
        "ok"
    }

@app.post('/build_bioreactor_opts_model', tags=["AI method"])
async def build_bioreactor_opts_model():
    """Return ..."""
    return {
        "ok"
    }


@app.post('/run_bioreactor_opts', tags=["AI method"])
async def run_bioreactor_opts():
    """Return ..."""
    return {
        "ok"
    }

################################################################################
# GPT
################################################################################

@app.post('/run_gpt_analysis', tags=["AI Modelling"])
async def run_gpt_analysis():
    """Return ..."""
    return {
        "ok"
    }
    
@app.post('/get_data_for_gpt_analysis', tags=["AI Modelling"])
async def get_data_for_gpt_analysis():
    """Return ..."""
    return {
        "ok"
    }

@app.post('/run_gpt_analysis', tags=["AI Modelling"])
async def run_gpt_analysis():
    """Return ..."""
    return {
        "ok"
    }

################################################################################
# Build Reports
################################################################################

@app.get('/build_gpt_summary_report', tags=["Final report"])
async def get_summary_report():
    """Return ..."""
    return {
        "ok"
    }

@app.post('/build_summary_report', tags=["Final report"])
async def build_summary_report():
    """Return ..."""
    return {
        "ok"
    }

################################################################################
# Get Reports
################################################################################

@app.get('/get_gpt_summary_report', tags=["Get report"])
async def get_summary_report():
    """Return ..."""
    return {
        "ok"
    }

@app.get('/get_summary_report', tags=["Get report"])
async def get_summary_report():
    """Return ..."""
    return {
        "ok"
    }


################################################################################
# Clear methods
################################################################################

@app.get('/clear_uploaded_data', tags=["Clear data"])
async def clear_uploaded_data():
    """Clear the data uploaded by the user during the session"""
    return {
        "ok"
    }

@app.get('/clear_generated_results', tags=["Clear data"])
async def clear_generated_results():
    """Clear the data generated from processes during the session"""
    return {
        "ok"
    }


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
