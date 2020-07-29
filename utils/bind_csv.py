import pandas as pd
import glob

path = '../../../FaceAlignment/LS3D-W/300VW-3D/CatA/'
path = '../../../FaceAlignment/LS3D-W/300VW-3D/CatB/'
path = '../../../FaceAlignment/LS3D-W/300VW-3D/CatC/'
path = '../../../FaceAlignment/LS3D-W/300VW-3D/Trainset/'
path = 'csv/'
csv_list = glob.glob(path + '*.csv')
print(csv_list)

df = pd.DataFrame()

for csv in csv_list:
    data = pd.read_csv(csv, index_col=0)
    print("shape data:", data.shape)
    df = df.append(data)

print(df.head())
print("shape df:", df.shape)
df.to_csv('preds_all.csv')
print("Predictions saved!")
