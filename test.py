# %%
import pandas as pd

# %%
version = '1.1.2'
filename = f'~/jlab/data/usdrub_{version}.csv'
#csv_path = '~/notebook_server/data/USDRUB.csv'
usdrub = pd.read_csv(filename)

version_save = '1.1.2+'
savefilename = f'~/jlab/result/usdrub_{version_save}.csv'
usdrub.to_csv(savefilename)

# %%
version = '1.1.2'
filename = f'~/jlab/data/usdrub_{version}.csv'
#csv_path = '~/notebook_server/data/USDRUB.csv'
usdrub = pd.read_csv(filename)

version_save = '1.1.2+'
savefilename = f'~/jlab/result/usdrub_{version_save}.csv'
usdrub.to_csv(savefilename)

# %%



