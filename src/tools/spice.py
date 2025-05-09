import subprocess # to run bash scripts from python 
import pandas as pd 
import tempfile # to manage temporary files (e.g. figures)
import zipfile # to zip up files 
import base64 # to handle byte objects (figure embedding to html)
import matplotlib.pyplot as plt

from pathlib import Path 
from importlib.resources import files
from .core import GROUP 
from .core import SAMPLE
from io import BytesIO # also to handle byte objects 

"""
Functions used for streamlining the initial processing steps e.g. parallelising reading files and job execution
"""
def LockFileGroup(group: list[str], directory: Path):
    """
    Given a list of samples and a directory, lock those files in place if they ALL exist in the directory. 
    Implements a soft locking since we're only checking for existence of files rather than active writing of files
    Allows for .tsv files in case we want to re-run or something 
    
    """
    lock_paths = [] 

    for sample in group: 
        bam_file = Path(f"{sample}.bam")
        bam_path = directory / bam_file 
        lock_bam = bam_path.with_suffix(bam_path.suffix + ".lock")

        tsv_file = Path(f"{sample}.tsv")
        tsv_path = directory / tsv_file 
        lock_tsv = tsv_path.with_suffix(tsv_path.suffix + ".lock")

        if (not bam_path.exists()) and (not tsv_path.exists()): 
            break # stop if neither file exists

        try: 
            if bam_path.exists(): # if bam path exists, touch the lock bam file 
                lock_bam.touch(exist_ok=False) 
                lock_paths.append(lock_bam)
            else: 
                lock_tsv.touch(exist_ok=False)
                lock_paths.append(lock_tsv)

        except FileExistsError:
            break # stop if lock file already exists 

    else: 
        return lock_paths
    
    # Clean up any partial locks if something failed
    for lock in lock_paths:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass

    return None

# Checks which groups have files in the current directory 
def SearchGroups(sample_dict: dict, group_IDs: list[str], directory: Path):
    """
    input: 
        - dictionary of group IDs and sample lists 
        - list of group IDs
    """
    locked_groups = [] 
    locked_paths = []

    for group_ID in group_IDs: 

        lock_files = LockFileGroup(sample_dict[group_ID], directory)

        if lock_files: 
            locked_groups.append(group_ID)
            locked_paths.extend(lock_files)
        else: 
            continue 
    
    return locked_groups, locked_paths

def RunBamReadcount(*args):
    # formatted like this so that it can be called in ProcessPoolExecutor.map()
    bam_path, out_dir, sites_path, ref_gnm_path = args 

    script = files("tools.data").joinpath("read_sites.sh")
    command = f"{str(script)} {bam_path} {out_dir} {sites_path} {ref_gnm_path}"
    subprocess.run(["bash", "-c", command], check=True)
    return None

def GenerateSampleObjects(*args):
    sample_sites_path = args[0]
    return SAMPLE(sample_sites_path)

def GenerateHTMLQC(figure_list, group_name, out_dir: Path):
    with tempfile.TemporaryDirectory() as temp_dir: 
        
        figure_data_urls = [] 

        for i, fig in enumerate(figure_list):
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)

            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            figure_data_url = f"data:image/png;base64,{img_base64}"
            figure_data_urls.append(figure_data_url)
        
        html_file = out_dir / f"{group_name}_QC.html"
        with open(html_file, 'w') as f:
            f.write("<html><body>\n")
            f.write(f"<h1>{group_name} QC Figures</h1>\n")
            for i, data_url in enumerate(figure_data_urls):
                # Embed each Base64-encoded image in the HTML
                f.write(f'<img src="{data_url}" alt="Figure {i + 1}"><br>\n')
            f.write("</body></html>\n")
        
    return None

def ZipFigures(figure_list, group_name: str, out_dir: Path):
    with tempfile.TemporaryDirectory() as temp_dir: 
        temp_dir = Path(temp_dir)
        fig_files = [] 
        for i, fig in enumerate(figure_list):
            fig_file = temp_dir / f"{group_name}_fig{i}.png"
            fig.savefig(fig_file, format='png')
            fig_files.append(fig_file)
        
        zip_file = out_dir / f"QC_plots_{group_name}.zip"
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf: 
            for fig_file in fig_files: 
                zipf.write(fig_file, fig_file.name)

            print(f"QC plots for {group_name} save in {zip_file}.")
    return None