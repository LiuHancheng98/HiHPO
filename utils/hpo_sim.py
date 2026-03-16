import pandas as pd
import numpy as np
import math
from collections import defaultdict
from scipy.sparse import dok_matrix, csr_matrix, save_npz
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from goatools.obo_parser import GODag
from ontology import HumanPhenotypeOntology

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load data
def load_data(hpo_index_file, annotation_file):
    logging.info("Loading HPO index and annotation files")
    hpo_index = pd.read_csv(hpo_index_file, sep='\t', header=None, names=['HPO', 'Index'])
    annotations = pd.read_csv(annotation_file, sep='\t', header=None, names=['Protein', 'HPO', 'Date'])
    return hpo_index, annotations

# Step 2: Create protein-HPO annotation matrix
def create_annotation_matrix(annotations, hpo_terms):
    logging.info("Creating protein-HPO annotation matrix")
    proteins = annotations['Protein'].unique()
    df_train_hpo_annotation = pd.DataFrame(0, index=proteins, columns=hpo_terms)
    for _, row in annotations.iterrows():
        if row['HPO'] in hpo_terms:
            df_train_hpo_annotation.loc[row['Protein'], row['HPO']] = 1
    return df_train_hpo_annotation

# Step 3: Calculate IC
def calculate_ic(df_train_hpo_annotation):
    logging.info("Calculating Information Content (IC)")
    total_protein = len(df_train_hpo_annotation.index)
    freq = df_train_hpo_annotation.sum(axis=0) / total_protein
    ic = -freq.apply(lambda x: math.log2(x) if x > 0 else 0)  # Handle zero frequency
    return ic

# Step 4: Get ancestors
def get_ancestors(ontology, hpo_terms):
    logging.info("Caching ancestors for HPO terms")
    # Get descendants of HP:0000118 (Phenotypic abnormality)
    target_term = 'HP:0000118'
    valid_terms = ontology.get_descendants([target_term]) | {target_term}
    ancestors = defaultdict(set)
    for term in hpo_terms:
        ancestors[term] = set(ontology.get_ancestors([term]) & valid_terms)  # Include term itself
    return ancestors

# Step 5: Similarity function
def ic_sim(term_a, term_b, ic, ancestors):
    if term_a == term_b:
        return 1.0
    ancestors_a = ancestors.get(term_a, {term_a})
    ancestors_b = ancestors.get(term_b, {term_b})
    common_ancestors = list(ancestors_a & ancestors_b)
    if not common_ancestors:
        return 0.0
    ic_mica = max(ic.get(anc, 0) for anc in common_ancestors if anc in ic)
    sim = (2 * ic_mica / (ic.get(term_a, 0) + ic.get(term_b, 0))) * (1 - 1 / (1 + ic_mica)) if (ic.get(term_a, 0) + ic.get(term_b, 0)) > 0 else 0
    return sim

# Step 6: Parallelized similarity computation with progress bar
def compute_similarity_chunk(pairs, ic, ancestors, total_pairs):
    results = []
    for i, term_a, j, term_b in tqdm(pairs, total=len(pairs), desc="Processing chunk", leave=False):
        sim = ic_sim(term_a, term_b, ic, ancestors)
        results.append((i, j, sim))
    return results

def compute_similarity_matrix(hpo_terms, hpo_to_idx, max_index, ic, ancestors, n_jobs=-1):
    logging.info("Computing similarity matrix")
    sim_matrix = dok_matrix((max_index + 1, max_index + 1), dtype=np.float32)
    
    # Create pairs with indices (upper triangle to exploit symmetry)
    pairs = [(hpo_to_idx[term_a], term_a, hpo_to_idx[term_b], term_b) 
             for term_a in hpo_terms for term_b in hpo_terms if hpo_to_idx[term_a] <= hpo_to_idx[term_b]]
    total_pairs = len(pairs)
    
    # Split pairs into chunks for parallel processing
    chunk_size = max(1, total_pairs // (n_jobs if n_jobs > 0 else 1))
    chunks = [pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
    
    # Parallel computation with tqdm
    with tqdm(total=total_pairs, desc="Computing similarities") as pbar:
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(compute_similarity_chunk)(chunk, ic, ancestors, total_pairs) for chunk in chunks
        )
        for chunk in results:
            pbar.update(len(chunk))
    
    # Fill sparse matrix
    for chunk in results:
        for i, j, sim in chunk:
            if sim > 0:  # Store only non-zero similarities
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Symmetry
    
    return sim_matrix

# Step 7: Save similarity matrix
def save_similarity_matrix(sim_matrix, hpo_index, output_file):
    logging.info("Saving similarity matrix")
    # Convert to CSR format for saving
    sim_matrix_csr = sim_matrix.tocsr()
    
    # Save as sparse matrix
    save_npz(output_file, sim_matrix_csr)
    
    # Optionally, save as dense CSV for inspection (if matrix is small)
    if sim_matrix.shape[0] < 100:  # Adjust threshold as needed
        indices = range(sim_matrix.shape[0])
        sim_df = pd.DataFrame(sim_matrix.toarray(), index=indices, columns=indices)
        sim_df.to_csv(output_file.replace('.npz', '.csv'))

# Main function
def main(hpo_index_file, annotation_file, ontology_file, output_file, n_jobs=-1):
    # Load data
    hpo_index, annotations = load_data(hpo_index_file, annotation_file)
    
    # Load ontology
    # Initialize the HPO ontology
    ontology = HumanPhenotypeOntology(ontology_file, version="202401")
    
    # Get HPO terms from annotations (filtered by index file)
    hpo_terms = [term for term in annotations['HPO'].unique() if term in hpo_index['HPO'].values]
    
    # Create mapping from HPO terms to indices
    hpo_to_idx = dict(zip(hpo_index['HPO'], hpo_index['Index']))
    
    # Get maximum index to size the matrix
    max_index = hpo_index['Index'].max()
    
    # Create annotation matrix
    df_train_hpo_annotation = create_annotation_matrix(annotations, hpo_terms)
    
    # Calculate IC
    ic = calculate_ic(df_train_hpo_annotation)
    
    # Get ancestors
    ancestors = get_ancestors(ontology, hpo_terms)
    
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(hpo_terms, hpo_to_idx, max_index, ic, ancestors, n_jobs)
    
    # Save results
    save_similarity_matrix(sim_matrix, hpo_index, output_file)
    
    logging.info("Computation completed")

if __name__ == "__main__":
    hpo_index_file = "../data/hpo_list.txt"
    annotation_file = "../data/temporal_train.txt"
    ontology_file = "../data/hp.obo"
    output_file = "../data/temporal/hpo_similarity.npz"
    n_jobs = -1  # Use all available cores
    main(hpo_index_file, annotation_file, ontology_file, output_file, n_jobs)