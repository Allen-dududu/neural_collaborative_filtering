import scipy.sparse as sp
import numpy as np
import pandas as pd
import os
class MyDataSet(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        if os.path.exists("NCF_test.csv") and os.path.exists("NCF_train.csv"):
            print("测试训练文件已划分好")           
        else:
            print("在线划分测试训练文件")  
            self.splitFile()

        self.trainMatrix = self.load_rating_file_as_matrix('NCF_train.csv')
        self.testRatings = self.load_rating_file_as_list("NCF_test.csv")
        # self.testNegatives = self.load_negative_file("NCF_train.csv")
        # assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
    # 获取测试评分
    # 用户的最新一条评分
    def load_rating_file_as_list(self, filename):
        ratingList = []
        ratingList =  pd.read_csv("NCF_test.csv")[['userId','movieID']].values
        return ratingList

    # 加载训练矩阵
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        users = pd.read_table(
        "users.dat",
        header=None,
        sep="::",
        names=["userId", "gender", "age", "Occupation", "zip-code"]
        )["userId"].tolist()
        items = pd.read_table(
        "movies.dat",
        header=None,
        sep="::",
        names=["movieID", "Title", "Genres"]
        )["movieID"].tolist()

        mat = sp.dok_matrix((max(users)+1, max(items)+1), dtype=np.float32)
        trainRatings =  pd.read_csv("NCF_test.csv")
        for index , row in trainRatings.iterrows(): 
            if (row['rate'] > 0):
                mat[row['userId'], row['movieID']] = 1.0 
        return mat
     # 划分测试训练文件
    def splitFile(self):
        ratings = pd.read_table(
            'ratings.dat',
            header = None,
            sep = "::",
            names = ["userId","movieID","rate","timestamp"]
        )

        # ratings_test = pd.DataFrame()
        # ratings_train = pd.DataFrame([],columns=["userId","movieID","rate","timestamp"])
        ratings_test = ratings.sort_values('timestamp', ascending=False).groupby('userId').first().reset_index()
        ratings.equals(ratings_test)
        ratings_train = ratings[ratings.eq('movieID') == False].dropna()

        ratings_test.to_csv("NCF_test.csv")
        ratings_train.to_csv("NCF_train.csv")
        print(ratings_test.head())
        print(ratings_train.head())

# 
# dataset = Dataset()