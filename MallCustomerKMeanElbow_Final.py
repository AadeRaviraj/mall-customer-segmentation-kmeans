import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    # - -----------------------------------------------------------
    # Step 1 : Load the dataset 
    # -------------------------------------------------------------
    print("Load the dataset ")
    df = pd.read_csv("Mall_Customers.csv")
    
    print("First few records --")
    print(df.head())
    print("Shape of dataset :")
    print(df.shape)
    
    
    print("Missing values : ")
    print(df.isnull().sum())
    
    
    # - -----------------------------------------------------------
    # Step 2 : Select feature (independent) 
    # -------------------------------------------------------------
    print("Step 2 : Select feature (independent) ")
    
    X = df[["AnnualIncome","SpendingScore"]]
    
    print("Selected feature : ")
    print(X.head())
    print("selected features : ")
    print(X.shape)
    
    
    
    # - -----------------------------------------------------------
    # Step 3 : Scale the data
    # -------------------------------------------------------------
    
    
    scaler = StandardScaler()
    X_Scale = scaler.fit_transform(X)
    
    print("Data after Scaling : ")
    
    print(X_Scale[:5])
    
    
    # - -----------------------------------------------------------
    # Step 4 : Use elbow method
    # -------------------------------------------------------------
    
    WCSS = []
    
    for i in range(1,11):
        model = KMeans(n_clusters=i,random_state= 42,n_init=10)
        model.fit(X_Scale)
        WCSS.append(model.inertia_)
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1,11),WCSS,marker ="o")
    plt.xlabel("Number of  cluster")
    plt.ylabel("WCSS")
    plt.title("Elbow method")
    plt.grid(True)
    # plt.show()

    
    # - -----------------------------------------------------------
    # Step 5 :Train the model
    # -------------------------------------------------------------
    
    
    model = KMeans(n_clusters=4,random_state=42,n_init=10)
    
    clusters  = model.fit_predict(X_Scale)
    
    df["Cluster"] = clusters
    
    print("Dataset with clusters : ")
    print(df.head(40))
    
    
    
if __name__ == "__main__":
    main()