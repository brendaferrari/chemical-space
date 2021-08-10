from lapjv import lapjv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import seaborn as sns

#Adapted from https://gist.github.com/iwatobipen/f8b0e8ea2c872e7ccf34ab472454ce6c#file-chemicalspace_lapjv-ipynb

# To read dataset
smi = pd.read_csv('dataset.csv')
print(smi)
print(smi.columns)

# To name variables
mols = [Chem.MolFromSmiles(smi) for smi in smi.smiles]
sampleid = smi.id
sampleidx = list(range(len(mols)))
samplemols = [mols[i] for i in sampleidx]
sampleact = [(smi['pic50'][idx]) for idx in sampleidx]

fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=4096) for m in samplemols]

# To define array
def fp2arr(fp):
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

X = np.asarray([fp2arr(fp) for fp in fps])
print(X.shape)

# To draw grid space (The data must be size^2) 
size = 9

N = size*size

# To calculate PCA and TSNE
data = PCA(n_components=81).fit_transform(X.astype(np.float32))
embeddings = TSNE(init='pca', random_state=794, verbose=2).fit_transform(data)
embeddings -= embeddings.min(axis=0)
embeddings /= embeddings.max(axis=0)

## To calculate grid space
grid = np.dstack(np.meshgrid(np.linspace(0,1,size), np.linspace(0,1,size))).reshape(-1,2)

print(embeddings.shape)

cost_mat = cdist(grid, embeddings, 'sqeuclidean').astype(np.float32)
cost_mat2 = cost_mat * (10000 / cost_mat.max())
print(cost_mat2.shape)
print(grid.shape)

row_asses, col_asses, _ = lapjv(cost_mat2)

grid_lap = grid[col_asses]

## To plot
#set size of graph
sns.set(rc={'figure.figsize':(30,30)})

# colormap and colorbar
color=sns.color_palette("coolwarm", as_cmap=True)
max_height = np.max(sampleact)   # get range of colorbars so we can normalize
min_height = np.min(sampleact)
rgba = [color((k-min_height)/max_height) for k in sampleact] 
norm = cm.colors.Normalize(vmin=min_height, vmax=max_height, clip=False)

# To set figure and scatterplot
sns.set(font_scale = 3)
sns.scatterplot(x=grid_lap[:,0], y=grid_lap[:,1], data=fps, hue=sampleact, palette=color, s=10000, edgecolor=None, legend=False, alpha=None)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color)).set_label("pIC50", size=30)

#To set text inside markers
for i, txt in enumerate(sampleid):
    plt.annotate(txt, (grid_lap[:,0][i], grid_lap[:,1][i]), fontsize=16, ha='center')
#####

plt.xlabel('Morgan Fingerprint', fontsize=35)
plt.ylabel('Morgan Fingerprint', fontsize=35)
plt.title('Chemical Space', fontsize=45, fontweight='bold')
plt.savefig("chemical_space_grid.png", bbox_inches='tight')