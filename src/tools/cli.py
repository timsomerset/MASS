import os 
import csv
import pickle
import argparse
import tempfile
import pandas as pd 
import matplotlib.pyplot as plt 

from filelock import FileLock # for writing to common .csv file 
from pathlib import Path 
from .core import GROUP 
from . import spice
from concurrent.futures import ProcessPoolExecutor # for parallel processing 

def Flag(args):
    # get number of available cores
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    # setting inputs as correct types 
    work_dir = Path(args.dir)
    assignment_sheet = Path(args.sample_sheet)
    sites_path = Path(args.snp_sheet)
    out_dir = Path(args.output)
    ref_gnm_path = Path(args.ref_genome)
    if args.temp_override: 
        temp_override = Path(args.temp_override)

    # do some argument cleaning and error handling here 
    #   |
    #   |
    #   |
    #   |
    #   _

    # read in the sample assignment sheet
    if assignment_sheet.suffix == ".csv": 
        assignment_df = pd.read_csv(assignment_sheet, sep=",", names = ["sample_ID", "group_ID"])
    elif assignment_sheet.suffix == ".tsv":
        assignment_df = pd.read_csv(assignment_sheet, sep="\t", names = ["sample_ID", "group_ID"])
    
    # convert to dictionary:
    #   group_ID:[sample_ID, ...]
    sample_dict = assignment_df.groupby('group_ID')['sample_ID'].apply(list).to_dict()
    group_IDs = sample_dict.keys()
    print(group_IDs)
    # generate list of group IDs whose samples all exist in the working directory
    workable_groups, locked_paths = spice.SearchGroups(sample_dict, group_IDs, work_dir)

    if len(workable_groups) == 0: 
        print(f"No group specified in {assignment_sheet} has a complete set of samples in {work_dir}.\n Job has been terminated.")
        if len(locked_paths) > 0: 
            for path in locked_paths: 
                path.unlink()
        return
    try:
        output_data = [] 
        # for each group, we: 
        #   - run bam-readcount on all the bam files 
        #   - generate SAMPLE objects 
        #   - generate a GROUP object 
        with tempfile.TemporaryDirectory() as tmpdir: # each instance of this script generates a new temp dir 
            if args.temp_override: 
                tmp_dir = temp_override
            else:
                tmp_dir = Path(tmpdir)
            for group in workable_groups: 
                # packaging arguments for inputting to RunBamReadcount 
                arg_list = [] 
                for sample in sample_dict[group]: 
                    arg_list.append(
                        (
                            work_dir / f"{sample}.bam", # bam path 
                            tmp_dir, # output directory - bash will output the .tsv files into the temp dir
                            sites_path, # sites path 
                            ref_gnm_path # ref genome path 
                        )
                    )

                # this enables *some* parallel processing depending on resource allocation 
                with ProcessPoolExecutor(max_workers=cores) as executor: 
                    executor.map(spice.RunBamReadcount, arg_list)
                # this does the following: 
                #   - checks if the bam file exists
                #   - defines the output .tsv file 
                #   - checks if output .tsv file already exists. If yes, ends job early. 
                #   - if no .tsv, check if .bai exists. If no, run samtools index 
                #   - with .bai file, runs bam-readcount 
                # this is run on all samples in the group before proceeding
                
                arg_list = [] 
                for sample in sample_dict[group]: 
                    arg_list.append(
                        (
                            tmp_dir / f"{sample}.tsv" # sites path - in temp dir 
                        )
                    )
                with ProcessPoolExecutor(max_workers=cores) as executor: 
                    samples = executor.map(spice.GenerateSampleObjects, arg_list)
                
                sample_list = list(samples) 

                # perform analysis 
                group_obj = GROUP(sample_list, ref_sites_path = sites_path)
                group_obj.CompareSamples()
                row, graph, barplot, scatterplot = group_obj.FlagOutliers()      
                row.insert(0, group) # adds group name to the row 
                
                dump_file = out_dir / f"{group}.pkl"
                pickle.dump(group, dump_file.open("wb"))

                # handle outputs 
                output_data.append(row)
                if graph: 
                    spice.ZipFigures(
                        [graph, barplot, scatterplot], 
                        group,
                        out_dir 
                        )
                plt.close(barplot)
                plt.close(graph)
                plt.close(scatterplot)
            
            out_csv_path = out_dir / "MASS_summary.csv"
            print(f"Writing output to: {str(out_csv_path)}")
            lock_csv = out_csv_path.with_suffix(out_csv_path.suffix + ".lock")

            lock = FileLock(lock_csv)

            with lock: 
                print("Lock obtained", flush=True)
                if not out_csv_path.is_file(): # adds header line if file doesn't exist 
                    with out_csv_path.open("w") as write_file:
                        writer = csv.writer(write_file)
                        writer.writerow([
                            "Patient_ID", 
                            "Flag", 
                            "Outlier_flag", 
                            "Uncertain_flag", 
                            "Outlier_samples", 
                            "Uncertain_samples"
                        ])
                        for row in output_data: 
                            writer.writerow(row)
                else: # otherwise append extra lines
                    with out_csv_path.open("a", newline="") as write_file:
                        writer = csv.writer(write_file)
                        for row in output_data: 
                            writer.writerow(row)
            
            print(f"{group} data written to {str(out_csv_path)}", flush=True)
            # clean up the locked files 
            if len(locked_paths) > 0: 
                for path in locked_paths: 
                    path.unlink()
    except:
        print("Some exception triggered")
        if len(locked_paths) > 0: 
            for path in locked_paths: 
                path.unlink()


def main():
    parser = argparse.ArgumentParser() 
    subparser = parser.add_subparsers(
        dest="command", required=True
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--dir', 
        default = "Testing/TestGroup",
        help="Path to .bam files")
    parent_parser.add_argument(
        '--sample_sheet',
        help="Path to .tsv or .csv file containing sample names in the first column (matching their file names) and \
            their patient label in the second column."
    )
    parent_parser.add_argument(
        '--snp_sheet', 
        default = "SNPData/somalier_sites.txt",
        help="Path to .tsv file for all sites being searched")
    parent_parser.add_argument(
        '--collect_data', 
        action="store_true", 
        help="Boolean for pickle dumping pval_mat, match_mat, overlap_mat and sample names"    
    )
    parent_parser.add_argument(
        '--output', 
        help="Directory to save outputs into."
    )
    parent_parser.add_argument(
        '--ref_genome', 
        help="Path to reference genome"
    )
    parent_parser.add_argument(
        '--temp_override', 
        default=None, 
        help="Path to alternative temp directory."
    )


    flag_parser = subparser.add_parser('flag', parents = [parent_parser])
    flag_parser.set_defaults(func=Flag)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__": 
    main() 