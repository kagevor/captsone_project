from data_loading import load_data
from data_preprocessing import data_preprocessing
import warnings
warnings.filterwarnings('ignore')

data = load_data()
data = data_preprocessing(data)
print(data.head())
