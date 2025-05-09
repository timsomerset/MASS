#%%
import os 
import ast
import warnings

import pandas as pd 
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import pygraphviz as pgv

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path 
from scipy.stats import binom
from scipy.stats import poisson_binom

#%%
# CR notes: 
# - error handling 
# - consistency with sample indices/names 

class SAMPLE:
    def __init__(self, sample_sites_path: Path):
        self.sites_dict = GenSampleSiteDict(sample_sites_path)
        self.name = self.sites_dict.get("sample_ID")
        self.sites = list(self.sites_dict.keys()) # position of reads 
        site_info = list(self.sites_dict.values()) # [read, depth] info per site
        self.reads = [info[0] for info in site_info] # read only 
        self.depth = [info[1] for info in site_info] # depth only 


class GROUP: 
    def __init__(self, SAMPLE_list: list[Path], **kwargs):
        '''
        input: 
            - list of SAMPLE objects belonging to the same group 
            Either:
            - ref_sites_path: Path object to reference sites file
            Or: 
            - ref_sites: self.ref_sites 
            - ref_meta: self.ref_meta
        sets: 
            - sample_names (list of sample names) 
            - ref_sites (list of sites in the reference sites file)
            - ref_meta (meta data on the reference sites)
            - read_matrix (populated read matrix) 
            - depth_matrix (populated depth matrix)
        '''
        self.sample_names = [sample.name for sample in SAMPLE_list]
        self.SAMPLE_list = SAMPLE_list

        # this allows us to easily generate new GROUP objects from a GROUP object   
        # i.e. we can do GROUP(new_SAMPLE_list, self.ref_sites, self.ref_meta)
        ref_sites_path = kwargs.get('ref_sites_path', None)
        ref_sites = kwargs.get('ref_sites', None)
        ref_meta = kwargs.get('ref_meta', None)

        if not ref_sites and not ref_meta: 
            #CR: breaks if ref_sites_path is not provided. Error message if missing!
            self.ref_sites, self.ref_meta = GenRefSitesMeta(ref_sites_path, len(SAMPLE_list))
        else: 
            self.ref_sites = ref_sites  
            self.ref_meta = ref_meta 

        read_matrix = np.empty(
            (len(self.sample_names), len(self.ref_sites)),
            dtype='U50'
        )
        # ref sites no longer contains sample_ID as the first entry!!!!!
        depth_matrix = np.zeros(read_matrix.shape)
        
        for i in range(read_matrix.shape[0]):
            read_matrix[i,:] = [SAMPLE_list[i].sites_dict.get(site, ["", 0])[0] for site in self.ref_sites]
            depth_matrix[i,:] = [SAMPLE_list[i].sites_dict.get(site, ["", 0])[1] for site in self.ref_sites]
        
        self.read_matrix = read_matrix 
        self.depth_matrix = depth_matrix

    def SingleCompare(self, sample1, sample2, method = "EPBH0Same"):
        """
        input: 
            - read matrix, depth matrix, ref sites, ref meta 
            - index of two samples 
            - comparison method (defaults to EPB, H0: samples are different)
        sets: 
            - none 
        gets: 
            - pval of comparison
            - n_overlap of comparison
            - n_match of comparison
        """
        if method == "EPBH0Diff":
            p_val, n_overlap, n_match, mean_match = EPBMultiRead(
                self.read_matrix, 
                self.depth_matrix, 
                self.ref_sites, 
                self.ref_meta, 
                sample1, 
                sample2
            )
            return p_val, n_overlap, n_match, mean_match
        elif method == "EPBH0Same": 
            p_val, n_overlap, n_match, mean_match = EPBMultiReadH0Same(
                self.read_matrix, 
                self.depth_matrix, 
                self.ref_sites, 
                self.ref_meta, 
                sample1, 
                sample2
            )
            return p_val, n_overlap, n_match, mean_match
        else: 
            warnings.warn(f"\'{method}\' is not a valid method for comparing samples. Please choose one of: \n -\'EPBH0Diff\'\n -\'EPBH0Same\'")
        
    def CompareSamples(self, method="EPBH0Same"):
        """ 
        input: 
            - method of comparison. Choose from: 
                - 'EPBH0Diff': exact poisson binomial, null hypothesis of samples are random 
                - 'EPBH0Same': exact poisson binomial, null hypothesis of samples are from the same person
        sets: 
            - pval matrix 
            - match matrix 
            - overlap matrix 
        gets: 
            - pval matrix 
            - match matrix 
            - overlap matrix
        """

        sample_names = self.sample_names

        # 1. Initialise the output matrices
        n_samples = len(sample_names)

        pval_mat = np.ones(
            (n_samples, n_samples)
        )
        overlap_mat = np.zeros(
            (n_samples, n_samples)
        )
        match_mat = np.zeros(
            (n_samples, n_samples)
        )
        mean_match_mat = np.zeros(
            (n_samples, n_samples)
        )

        # 2. Perform comparisons to populate output matrices 
        upper_tri_indices = [[i,j] for i in range(n_samples) for j in range(n_samples) if i < j]
        for coord in upper_tri_indices: 
            pval, overlaps, matches, mean_match = self.SingleCompare(
                coord[0], 
                coord[1], 
                method=method
            )

            pval_mat[coord[0], coord[1]] = pval

            overlap_mat[coord[0], coord[1]] = overlaps

            match_mat[coord[0], coord[1]] = matches 

            mean_match_mat[coord[0], coord[1]] = mean_match 

        self.pval_mat = pval_mat 
        self.match_mat = match_mat 
        self.overlap_mat = overlap_mat
        self.mean_match = mean_match_mat
    
    def FlagOutliers(self):
        """
        Function used to identify outliers after running CompareSamples
        input: 
            - pathlib object to file for saving QC plots 
        set: 
            - none 
        get: 
            - list that can be added to row of .tsv file containing: 
                - group name 
                - binary for whether outliers are detected 
                - list of detected outliers 
            - QC plots if outliers exist 
        """
        try: 
            outlier_names = []
            uncertain_names = []
            for i, name in enumerate(self.sample_names):
                indices = [[i,j] if i < j else [j,i] for j in range(len(self.sample_names))] # indices in pval mat for this sample
                pvals = [self.pval_mat[index[0], index[1]] for index in indices if index[0] != index[1] ] # remove self-comparison (pval there defaults to 1)
                count_same = np.sum(np.array(pvals) > 0.05)
                count_outlier = np.sum(np.array(pvals) < 0.01)
                count_uncertain = np.sum((np.array(pvals) < 0.01) & (np.array(pvals) > 0.05))

                # Classification of sample 
                if (count_uncertain < 0.3*len(pvals)) and (count_outlier > count_same): # low uncertainty, strong dominance of outliers
                    outlier_names.append(name)
                elif count_uncertain > 0.5*len(pvals): # high uncertainty 
                    uncertain_names.append(name)
                elif count_uncertain > 0.3 and count_uncertain < 0.5 and (count_outlier >= count_same): # some uncertainty, leaning towards outlier
                    uncertain_names.append(name)
            
            outlier_flag = 0 
            uncertain_flag = 0
            # Classification of group
            if len(outlier_names) > 0 and len(uncertain_names) == 0:
                flag = 2 # outlier
                outlier_flag = 1
            elif len(outlier_names) >= 0 and len(uncertain_names) > 0:
                flag = 1 # some uncertainty 
                uncertain_flag = 1
                if len(outlier_names) > 0: 
                    outlier_flag = 1
            else: 
                flag = 0 # safe 
            
            # output data
            row = [
                flag, 
                outlier_flag, 
                uncertain_flag, 
                outlier_names, 
                uncertain_names
            ]
            
            if flag > 0: # do QC if flagged
                graph = self.GenGraph(show=False, )
                barplot, scatterplot = RunQC(
                    self.sample_names, 
                    self.overlap_mat, 
                    self.mean_match, 
                    self.match_mat, 
                    self.pval_mat
                )
                return (row, graph, barplot, scatterplot)
            else: 
                return (row, None, None, None)
        
        except AttributeError:
            raise Exception("CompareSamples must be run before FlagOutliers")  # test if CompareSamples has been run 

    def GenGraph(self, save_path=None, min_overlap=0, show=False):
        """
        input: 
            - path to save graph to 
            - boolean for significant connections only 
            - minimum overlap number (for quality filtering)
            - boolean for printing figure immediately
        set: 
            - none 
        get: 
            - figure 
        """
        sample_names = self.sample_names
        n_samples = len(sample_names)
    
        G = nx.Graph() 

        # build graph, set edge weights 
        for sample in sample_names: 
            G.add_node(sample)
        for i in range(n_samples): 
            for j in range(i+1, n_samples): 
                if self.overlap_mat[i,j] >= min_overlap:
                    G.add_edge(sample_names[i], sample_names[j], weight=(self.pval_mat[i,j])) # normal p-val b/c small p-val => difference
                else: 
                    continue
        
        # set node values 
        node_medians = {}
        for node in G.nodes:
            # Get all edge weights connected to the node
            edges = G.edges(node, data=True)
            weights = [data['weight'] for _, _, data in edges] # no correction necessary

            # Calculate the median edge weight for the node
            node_medians[node] = np.mean(weights) if weights else 0  # Default to 0 if no edges

        # define colour 
        boundaries = [0.0, 0.01, 0.049, 0.05, 1.0]
        colors = ["purple", "red", "xkcd:saffron", "yellow", "xkcd:pale yellow"] # colours are reversed b/c interpretation of p-value reversed 
        cmap = LinearSegmentedColormap.from_list("map", list(zip(boundaries, colors)))
        weights = nx.get_edge_attributes(G, 'weight')
        edge_colors = [cmap(weight) for weight in weights.values()] # again, don't invert p-val
                
        node_colors = [cmap(node_medians[node]) for node in G.nodes]
        pos = nx.forceatlas2_layout(G, weight='weight')#, scaling_ratio=0.5)

        fig, ax = plt.subplots(figsize=(13, 13))
        fig.tight_layout(pad = 5)

        nx.draw(
            G, pos, with_labels=True, node_size=500, node_color=node_colors, 
            edge_color=edge_colors, font_size=10, ax = ax
        )

        ax.set_title("Graph Representation of Pairwise Comparisons")

        if save_path is not None:
            if isinstance(save_path, Path):
                fig.savefig(save_path.open("wb"), dpi=300)
            else:
                warnings.warn("Path for saving plot is not a Pathlib object. Plot not saved.")

        if show:
            plt.show() 
        
        return fig

    def GenAltGroup(self, SAMPLE_object, mask=None):
        # allow sample indices for mask 
        """
        Input: 
            - a SAMPLE object to introduce to this group 
            - mask: a list of sample names to omit from this new group
        sets: 
            - none 
        gets: 
            - a new GROUP object with the different samples 
        """
        if mask:
            samples_include = list(set(self.sample_names) - set(mask))
            include_indices = [self.sample_names.index(sample) for sample in samples_include]
            SAMPLE_list = [self.SAMPLE_list[i] for i in range(len(self.sample_names)) if i in include_indices]
            SAMPLE_list.append(SAMPLE_object)
        else: 
            SAMPLE_list=self.SAMPLE_list
            SAMPLE_list.append(SAMPLE_object)

        new_group = GROUP(SAMPLE_list, ref_sites=self.ref_sites, ref_meta=self.ref_meta)
        return new_group

    def ExtractSample(self, sample_name):
        sample_index = self.sample_names.index(sample_name)
        return self.SAMPLE_list[sample_index]
        
     
# used in SAMPLE class __init__
def ParseSampleSite(row):
    """ 
     type: internal (this is used in other functions and not called as a method or individual function)
     input: row of a pandas df containing sites information for a sample
     output: dictionary entry that can be appended to a dictionary
     """
    row = row.to_list()

    chrom = row[0]
    pos = str(row[1])
    key=f"{chrom}:{pos}"
    wt = row[2].upper()

    qc_list = row[5:11]
    read_counts = []
    reads = ["A", "C", "G", "T"]
    for qc in qc_list: 
        if isinstance(qc, str):
            split = qc.split(":")
            nucl = split[0]
            if nucl.upper() == "N":
                continue
            else:
                count = ast.literal_eval(split[1])
                read_counts.append(count)
    n_reads = np.sum(read_counts)
    mut_reads = [reads[i] for i in range(len(reads)) if (reads[i] != wt) and (read_counts[i] > 0) ]
    if len(mut_reads) == 0: 
        read = wt 
    else: 
        read = mut_reads[0]

    return {key: [read, n_reads]}

# used in SAMPLE class __init__
def GenSampleSiteDict(sample_sites_path, within=None):
    """ 
     type: external
     input: 
        - sample_sites_path: Pathlib object to tsv output of bam-readcount
        - within: a dictionary of reference sites that the sample sites should be contained in
     output: Dictionary to lookup values at each observed site in sample 
     """
    path = sample_sites_path 
    if not path.is_file(): 
        if path.is_dir(): 
            raise Exception("Invalid path to site file: path is directory ")
        else: 
            raise Exception("Invalid path to site file: file does not exist")
    elif(path.suffix != ".tsv"): 
            raise Exception("Invalid path to site file: file must be .tsv")
    else: 
        sites_colnames = [
            "chrom", 
            "pos", 
            "wt", 
            "reads", 
            "qc1", 
            "qcA", 
            "qcC", 
            "qcG", 
            "qcT", 
            "qcN", 
            "qcDel"
        ]
        sites_df = pd.read_csv(
            path, 
            sep="\t", 
            header=None,
            engine="c", 
            names=sites_colnames
        )
        sample_name = path.stem
        sample_dict = {
            "sample_ID": sample_name
        }
        dict_list = sites_df.apply(ParseSampleSite, axis=1).to_list()
        
        if within != None: 
            ref_sites = within.keys()
            for d in dict_list: 
                if list(d.keys())[0] in ref_sites: 
                    sample_dict.update(d)
        else: 
            for d in dict_list: 
                sample_dict.update(d)            

        return sample_dict

# used in GROUP class __init__
def GenRefSitesMeta(ref_sites_path, n_samples):
    """ 
     type: external
     input:
        ref_sites_path:     pathlib object pointing to SNP panel file (file compatible 
                            with bam-readcount)
        group_dir:          pathlib object pointing to directory of sample site files
                            for the label group
     output: 
        1. A NxM matrix (N = number of samples, M = number of sites in SNP panel)
        2. Ordered list of sites (as they will be inputted into the matrix)
        3. Dictionary with site as key, and [ref, alt, VAF] as value

        ref_sites no longer contains "sample_ID" as first entry!!!!!!!!!!
       """
    
    ref_sites_colnames = [
        "chrom", 
        "pos", 
        "end", 
        "ref", 
        "alt", 
        "vaf"
    ]
    ref_sites_df = pd.read_csv(
        ref_sites_path, 
        sep = "\t", 
        engine = "c", 
        names = ref_sites_colnames,
        header = None
    )
    ref_sites_df["pos"] = ref_sites_df["pos"].astype("Int64")
    ref_sites_df["end"] = ref_sites_df["end"].astype("Int64")

    # Generate site list
    site_query_list = ref_sites_df.apply(
        lambda x: f"{x.iloc[0]}:{str(x.iloc[1])}",
        axis=1
    ).to_list()

    # Generate matrix 

    ref_site_dict_list = ref_sites_df.apply(
        lambda x: {
            f"{x.iloc[0]}:{x.iloc[1]}":
            [x.iloc[3], x.iloc[4], x.iloc[5]]}, 
        axis=1
    )
    ref_site_dict = {} 
    for d in ref_site_dict_list: 
        ref_site_dict.update(d)


    return site_query_list, ref_site_dict


def RunQC(samples, overlap_mat, e_match, match, pval):
    """
    input: 
        - sample name list 
        - overlap matrix 
        - mean matrix 
        - match matrix 
        - pval matrix 
    output: 
        - QC barplot 
        - QC scatterplot
    """    
    # define colours
    boundaries = [0.0, 0.01, 0.049, 0.05, 1.0]
    colors = ["purple", "red", "xkcd:saffron", "yellow", "xkcd:pale yellow"]
    #colors = ["xkcd:pale yellow", "xkcd:saffron", "xkcd:saffron", "red", "purple"]
    cmap = LinearSegmentedColormap.from_list("map", list(zip(boundaries, colors)))

    upper_tri_indices = [[i,j] for i in range(len(samples)) for j in range(len(samples)) if i < j]
    overlaps = [overlap_mat[index[0], index[1]] for index in upper_tri_indices]
    pvals = [pval[index[0], index[1]] for index in upper_tri_indices]

    # Barplot 
    overlap_smallp = [overlaps[i] for i in range(len(overlaps)) if pvals[i] < 0.01]
    overlap_midp = [overlaps[i] for i in range(len(overlaps)) if pvals[i] > 0.01 and pvals[i] < 0.05]
    overlap_bigp = [overlaps[i] for i in range(len(overlaps)) if pvals[i] > 0.05]
    colours = ["purple", "xkcd:saffron", "xkcd:pale yellow"]

    bins = np.floor(np.max(overlaps)/20).astype(int) # bins of width 20

    barplot, ax = plt.subplots() 
    ax.hist([overlap_smallp, overlap_midp,  overlap_bigp ], bins, stacked=True, color=colours)
    ax.set_title("Distribution of Number of Overlaps")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Number of Overlaps")
    ax.legend({"p < 0.01": "purple", "0.01 < p < 0.05": "xkcd:saffron", "p > 0.05":"xkcd:pale yellow"})
    ax.set_xlim(0, np.max(overlaps) + 50)

    # Scatter plot
    expected_matches = [e_match[index[0], index[1]] for index in upper_tri_indices]
    matches = [match[index[0], index[1]] for index in upper_tri_indices]
    min_val = min(0, 0)  # Get the min range
    max_val = max(np.max(expected_matches), np.max(matches))  # Get the max range

    scatterplot, ax = plt.subplots()

    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="y = x", alpha = 0.4)

    sc = ax.scatter(np.array(expected_matches), np.array(matches), c = pvals, alpha = 0.5, cmap = cmap)
    ax.set_xlim(0, np.max(expected_matches)+20)
    ax.set_ylim(0, np.max(matches)+20)
    ax.set_xlabel("Expected Number of Matches")
    ax.set_ylabel("Actual Number of Matches")
    scatterplot.colorbar(sc, ax=ax, label="p-value")

    return barplot, scatterplot

# used in GROUP class SingleCompare
def EPBMultiRead(read_matrix, depth_matrix, ref_sites, ref_meta, sample1, sample2): # read & depth matrix era

    read_mat = read_matrix[[sample1, sample2],:]
    depth_mat = depth_matrix[[sample1, sample2],:]
    
    overlap_indices = np.where((read_mat[0] != "") & (read_mat[1] != ""))[0] # matrix coordinates for overlapping SNPs
    overlap_positions = [ref_sites[i] for i in overlap_indices] # genomic coordinates of the overlapped SNP sites
    overlap_meta = [ref_meta[x] for x in overlap_positions] # get VAF for these sites 
    VAFs = [meta[2] for meta in overlap_meta]
    
    read_mat = read_mat[:,overlap_indices] # restrict these to only overlapping sites 
    depth_mat = depth_mat[:,overlap_indices]

    T1 = [1 - binom.cdf(0, depth, 0.5) for depth in list(depth_mat[0])] # probability of mut read given depth in sample 1
    T2 = [1 - binom.cdf(0, depth, 0.5) for depth in list(depth_mat[1])] # probability of mut read given depth in sample 2
    
    n_match = len(np.where( (read_mat[0] == read_mat[1]) & (read_mat[1] != "") )[0])
    
    p_match = [p**2*t1*t2 + p**2*(1-t1)*(1-t2) + p*(1-t1)*(1-p) + (1-p)*p*(1-t2) + (1-p)**2 for p, t1, t2 in zip(VAFs, T1, T2) ]

    pbin = PoiBin(p_match)
    p_val = 1 - pbin.cdf(n_match)
    bounded_p_val = np.max([p_val, 0]) # numerical errors make the cdf value > 1 for high number of matches

    return bounded_p_val, len(overlap_indices), n_match, np.sum(p_match)

def EPBMultiReadH0Same(read_matrix, depth_matrix, ref_sites, ref_meta, sample1, sample2):
    read_mat = read_matrix[[sample1, sample2],:]
    depth_mat = depth_matrix[[sample1, sample2],:]
    
    overlap_indices = np.where((read_mat[0] != "") & (read_mat[1] != ""))[0] # matrix coordinates for overlapping SNPs
    overlap_positions = [ref_sites[i] for i in overlap_indices] # genomic coordinates of the overlapped SNP sites
    overlap_meta = [ref_meta[x] for x in overlap_positions] # get VAF for these sites 
    #VAFs = [np.min([meta[2], 1-meta[2]]) if meta[2] > 0.6 else meta[2] for meta in overlap_meta] # method 1
    VAFs = [meta[2] for meta in overlap_meta] # method 2
    
    read_mat = read_mat[:,overlap_indices] # restrict these to only overlapping sites 
    depth_mat = depth_mat[:,overlap_indices]

    T1 = [1 - binom.cdf(0, depth, 0.5) for depth in list(depth_mat[0])] # probability of mut read given depth in sample 1
    T2 = [1 - binom.cdf(0, depth, 0.5) for depth in list(depth_mat[1])] # probability of mut read given depth in sample 2
    
    n_match = len(np.where( (read_mat[0] == read_mat[1]) & (read_mat[1] != "") )[0])
    
    #p_match = [(1-p) + p*(t1*t2 + (1-t1)*(1-t2)) - 0.01 for p, t1, t2 in zip(VAFs, T1, T2) ] # method 1
    p_match = [(1-p)**2 + p**2 + 2*p*(1-p)*t1*t2 + 2*p*(1-p)*(1-t1)*(1-t2) for p, t1, t2 in zip(VAFs, T1, T2)] # method 2

    p_val = poisson_binom.cdf(n_match, p_match)
    bounded_p_val = np.min([p_val, 1]) # numerical errors make the cdf value > 0 for high number of matches

    return bounded_p_val, len(overlap_indices), n_match, np.sum(p_match)

def ExactPoiBin(sample_site_dicts, ref_site_dict, dict_1, dict_2): # sample dict era 
    """ 
    type: internal
    action: 
        - compares two samples using exact poisson binomial only counting matching sites
    input: 
        - all the ref site metadata 
        - populated sites matrix
        - row indices of the samples we want to compare
    output: 
        - p-value for the comparison
        - if collect == TRUE: 
            - number of overlapping sites 
            - number of matching sites 
        """
    
    sample1 = sample_site_dicts[dict_1]
    sample2 = sample_site_dicts[dict_2]

    all_sites1 = set(sample1.keys())
    all_sites2 = set(sample2.keys())
    
    overlap_sites = list(all_sites1.intersection(all_sites2))
    overlap_sites.remove("sample_ID")

    read_matrix = np.empty(
        (2, len(overlap_sites)),
        dtype='U50'
    )
    # depth_matrix = np.zeros(
    #     (2, len(overlap_sites))
    # )

    for i, site in enumerate(overlap_sites):
        samp1site = sample1.get(site)
        samp2site = sample2.get(site)

        read_matrix[0,i] = samp1site[0]
        read_matrix[1,i] = samp2site[0] 

        # depth_matrix[0,i] = samp1site[1]
        # depth_matrix[1,i] = samp2site[1]

   
    overlap_meta = [ref_site_dict.get(x) for x in overlap_sites] # get VAF for these sites 
    VAFs = [meta[2] for meta in overlap_meta]

    n_match = len(np.where( (read_matrix[0] == read_matrix[1]) & (read_matrix[1] != "") )[0])
    #p_match = [p**2*0.25 + (1-p)**2*0.25 for p in VAFS]
    p_match = [(1-p)**2 + (1-p)*p + 0.25*p**2 + 0.25*p**2 for p in VAFs]

    pbin = PoiBin(p_match)
    p_val = 1 - pbin.cdf(n_match)
    bounded_p_val = np.max([p_val, 0]) # numerical errors make the cdf value > 1 for high number of matches

    return bounded_p_val, len(overlap_sites), n_match

def ExactPoiBinMultiRead(sample_site_dicts, ref_site_dict, dict_1, dict_2): # sample dict era 
    """ 
    type: internal
    action: 
        - compares two samples using exact poisson binomial only counting matching sites
        - includes correction for probability of mutation given number of reads
    input: 
        - all the ref site metadata 
        - populated sites matrix
        - row indices of the samples we want to compare
    output: 
        - p-value for the comparison
        - if collect == TRUE: 
            - number of overlapping sites 
            - number of matching sites 
        """
    
    sample1 = sample_site_dicts[dict_1]
    sample2 = sample_site_dicts[dict_2]

    all_sites1 = set(sample1.keys())
    all_sites2 = set(sample2.keys())
    
    overlap_sites = list(all_sites1.intersection(all_sites2))
    overlap_sites.remove("sample_ID")

    read_matrix = np.empty(
        (2, len(overlap_sites)),
        dtype='U50'
    )
    depth_matrix = np.zeros(
        (2, len(overlap_sites))
    )

    for i, site in enumerate(overlap_sites):
        samp1site = sample1.get(site)
        samp2site = sample2.get(site)

        read_matrix[0,i] = samp1site[0]
        read_matrix[1,i] = samp2site[0] 

        depth_matrix[0,i] = samp1site[1]
        depth_matrix[1,i] = samp2site[1]

   
    overlap_meta = [ref_site_dict.get(x) for x in overlap_sites] # get VAF for these sites 
    VAFs = [meta[2] for meta in overlap_meta]

    n_match = len(np.where( (read_matrix[0] == read_matrix[1]) & (read_matrix[1] != "") )[0])
    #p_match = [p**2*0.25 + (1-p)**2*0.25 for p in VAFS]
    T1 = [1- binom.cdf(0, reads, 0.5) for reads in list(depth_matrix[0,:])]
    T2 = [1-binom.cdf(0, reads, 0.5) for reads in list(depth_matrix[1,:])]
    p_match = [p**1*t1*t2 + p**2*(1-t1)*(1-t2) + p*(1-t1)*(1-p) + (1-p)*p*(1-t2) + (1-p)**2 for p, t1, t2 in zip(VAFs, T1, T2) ]


    pbin = PoiBin(p_match)
    p_val = 1 - pbin.cdf(n_match)
    bounded_p_val = np.max([p_val, 0]) # numerical errors make the cdf value > 1 for high number of matches

    return bounded_p_val, len(overlap_sites), n_match

def DrawSubgraphOnNode(G, node):
    # Specify the target node
    target_node = node

    subgraph_nodes = [target_node] + list(G.neighbors(target_node))
    subgraph = G.subgraph(subgraph_nodes)

    pos = nx.forceatlas2_layout(subgraph, weight = 'weight')
    
    plt.figure(figsize=(12, 10))
    nx.draw(subgraph, pos, with_labels=True, edge_color="powderblue", node_color="violet")

    # Highlight the target node
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[target_node], node_color="pink", node_size=500)

    # Show the plot
    plt.show()

# %%
def GetPairCounts(sample_site_dicts, ref_site_dict, dict_1, dict_2):

    sample1 = sample_site_dicts[dict_1]
    sample2 = sample_site_dicts[dict_2]

    all_sites1 = set(sample1.keys())
    all_sites2 = set(sample2.keys())
    
    overlap_sites = list(all_sites1.intersection(all_sites2))
    overlap_sites.remove("sample_ID")

    read_matrix = np.empty(
        (2, len(overlap_sites)),
        dtype='U50'
    )
    depth_matrix = np.zeros(
        (2, len(overlap_sites))
    )

    for i, site in enumerate(overlap_sites):
        samp1site = sample1.get(site)
        samp2site = sample2.get(site)

        read_matrix[0,i] = samp1site[0]
        read_matrix[1,i] = samp2site[0] 

        depth_matrix[0,i] = samp1site[1]
        depth_matrix[1,i] = samp2site[1]

   
    overlap_meta = [ref_site_dict.get(x) for x in overlap_sites] # get VAF for these sites 
    VAFs = [meta[2] for meta in overlap_meta]
    mut_reads = [meta[1] for meta in overlap_meta]
    wt_reads = [meta[0] for meta in overlap_meta]


    n_mut_match = len(np.where( (read_matrix[0] == read_matrix[1]) & (read_matrix[1] == mut_reads) )[0])
    n_wt_match = len(np.where( (read_matrix[0] == read_matrix[1]) & (read_matrix[1] == wt_reads) )[0])
    n_mismatch = len(overlap_sites) - n_mut_match - n_wt_match

    #p_match = [p**2*0.25 + (1-p)**2*0.25 for p in VAFS]
    T1 = [1- binom.cdf(0, reads, 0.5) for reads in list(depth_matrix[0,:])]
    T2 = [1-binom.cdf(0, reads, 0.5) for reads in list(depth_matrix[1,:])]
    p_match = [p**1*t1*t2 + p**2*(1-t1)*(1-t2) + p*(1-t1)*(1-p) + (1-p)*p*(1-t2) + (1-p)**2 for p, t1, t2 in zip(VAFs, T1, T2) ]
    p_match = np.mean(p_match)
    p_mu = np.mean([p**2*t1*t2 for p, t1, t2 in zip(VAFs, T1, T2)])
    p_wt = np.mean([p**2*(1-t1)*(1-t2) + p*(1-t1)*(1-p) + (1-p)*p*(1-t2) + (1-p)**2 for p, t1, t2 in zip(VAFs, T1, T2)])


    return n_wt_match, n_mut_match, n_mismatch, p_match, p_mu, p_wt

def GetMutWtMisCounts(sample_site_dicts, ref_site_dict):
    # 1. Initialise the output matrices
    n_samples = len(sample_site_dicts)
    mut_matches = [] 
    wt_matches = []
    mis_matches = [] 
    p_matches = []
    p_mut_matches = []
    p_wt_matches = []

    # 3. Perform comparisons to populate output matrices 
    upper_tri_indices = [[i,j] for i in range(n_samples) for j in range(n_samples) if i < j]
    for coord in upper_tri_indices:
        n_wt_match, n_mut_match, n_mismatch, p_match, p_mu, p_wt = GetPairCounts(sample_site_dicts, ref_site_dict, coord[0], coord[1])
        mut_matches.append(n_mut_match)
        wt_matches.append(n_wt_match) 
        mis_matches.append(n_mismatch) 
        p_matches.append(p_match)
        p_mut_matches.append(p_mu)
        p_wt_matches.append(p_wt)
    
    return mut_matches, wt_matches, mis_matches, p_matches,  p_mut_matches, p_wt_matches

