===============================================================================
                        How to run
===============================================================================

Running the program is very simple. If you want to run it from command promt, 
please type "python3 vsm.py" after setting the user query or default query mode
along with the desired number of ranked output for every rank



===============================================================================
                        Important functions and variables
===============================================================================
document_dir: contains original data file
save_to: this directory contains the redefined documents from the document_dir, 
keeping only the description

user_query_processing: This function handles the cutom query input 
interactively from the console. The result is not saved in the file, but 
displayed in the console for convenience. The prompt stays alive until a user
hits enter or empty query.


default_query_processing: This function takes a query file as an input and
processes that using tokenize, stemming steps to feed into the ranking function.
The output file contains top-K ranked results for every query.

cosine similarity: 
sim = Q.A / norm(Q)*norm(A) where Q is the query vector and A is tf-idf matrix

in our case, Q.A = vec_dot = np.dot(q_mat.q, A['doc_'+str(i)])
and norm(Q)*norm(A) = vec_norm = (LA.norm(q_mat.q) * LA.norm(A['doc_'+str(i)]))
