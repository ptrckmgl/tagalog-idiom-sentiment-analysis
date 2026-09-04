from datasets import load_dataset

# Load a dataset and print the first example in the training set
newsph_dataset = load_dataset('C:\\Users\\patri\\Downloads\\newsph\\newsph\\')
print(newsph_dataset['train'])