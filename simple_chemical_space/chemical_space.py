from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
import numpy as np

#Adapted from https://gist.github.com/iwatobipen/f8b0e8ea2c872e7ccf34ab472454ce6c#file-chemicalspace_lapjv-ipynb

# To read dataset
smi = pd.read_csv('resources/dataset/dataset.csv')
print(smi)
print(smi.columns)

# To name variables
mols = [Chem.MolFromSmiles(smi) for smi in smi.smiles]
sampleid = smi.id
sampleidx = list(range(len(mols)))
samplemols = [mols[i] for i in sampleidx]
sampleact = [(smi['pic50'][idx]) for idx in sampleidx]
# print(sampleact)

# Fingerprint used: lfcfp6
fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=16384) for m in samplemols]

# To define array
def fp2arr(fp):
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

X = np.asarray([fp2arr(fp) for fp in fps])
print(X.shape)

# To calculate PCA and TSNE
data = PCA(n_components=94).fit_transform(X.astype(np.float32))
embeddings = TSNE(init='pca', random_state=794, verbose=2).fit_transform(data)
embeddings -= embeddings.min(axis=0)
embeddings /= embeddings.max(axis=0)

## To plot
# To set colormap and colorbar
color=sns.color_palette("coolwarm", as_cmap=True)
max_height = np.max(sampleact)   # get range of colorbars so we can normalize
min_height = np.min(sampleact)
rgba = [color((k-min_height)/max_height) for k in sampleact] 
norm = cm.colors.Normalize(vmin=min_height, vmax=max_height, clip=False)

# To set figure and scatterplot
sns.set(rc={'figure.figsize':(11,8)}, font_scale = 1)
color=sns.color_palette("Spectral", as_cmap=True)
g=sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], data=fps, hue=sampleact, palette=color, size=sampleact, edgecolor='black', legend=False, alpha=0.7)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color)).set_label("pIC50", size=14)

plt.xlabel('Morgan Fingerprint', fontsize=14)
plt.ylabel('Morgan Fingerprint', fontsize=14)
plt.title('Chemical Space', fontsize=20, fontweight='bold')
plt.savefig("chemical_space.png", bbox_inches='tight')