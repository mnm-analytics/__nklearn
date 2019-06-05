class Node:
    
    def __init__(self, X, y, label=None, max_depth=2, min_sample_split=1):
        self.X, self.y = X.copy(), y.copy()
        self.N = X.shape[0]
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.label = y.mean() if label == None else label
        self.left  = None
        self.right = None
        self.depth = None
        
        # 分岐パラメータ
        self.gini = 10e100
        self.column = None
        self.threshold = None

    def get_gini(self, t, f, y):
        '''
        2分岐後のジニ不純度を求める関数
        '''
        n_t, n_f = t.shape[0], f.shape[0]

        # 各ノードの目的変数の平均値を算出する。
        t_label, f_label = y[t.index].mean(), y[f.index].mean()

        # 各ノードのジニ係数を算出する。
        t_gini, f_gini = t_label*(1 - t_label), f_label*(1 - f_label)

        # 各ジニ係数のデータ数に応じた加重平均をとる
        gini = (n_t*t_gini + n_f*f_gini)/self.N
        return gini, t_label, f_label
        
    def spliter(self, X, y, min_sample_split=1):
        '''
        ジニ不純度が最小となる列と分岐点を求める関数
        '''
        for tgt_col in X.columns:
            vals = set(X[tgt_col].apply(int))
            for v in vals:
                cond = X[tgt_col] < v
                t, f = X[cond], X[~cond]                
                if t.shape[0] < min_sample_split or f.shape[0] < min_sample_split:
                    continue

                gini, _, _ = self.get_gini(t, f, y)
                
                if gini < self.gini:
                    self.gini = gini
                    self.column = tgt_col
                    self.threshold = v

    def split_node(self, depth=1):
        '''
        再帰的に、分割したデータをNodeクラスへと入力し分割する関数
        '''
        self.depth = depth
        
        if self.depth >= self.max_depth:
            return
        
        self.spliter(self.X, self.y, self.min_sample_split)
        if self.threshold == None:
            return

        cond = self.X[self.column] < self.threshold
        t, f = self.X[cond], self.X[~cond]
        gini, t_label, f_label = self.get_gini(t, f, self.y)
        
        self.left  = Node(X=t, y=self.y[t.index], label=t_label, max_depth=self.max_depth)
        self.right  = Node(X=f, y=self.y[f.index], label=f_label, max_depth=self.max_depth)
        self.left.split_node(depth + 1)   # recursive call
        self.right.split_node(depth + 1)  # recursive call

    def get_label(self, x):
        '''
        特徴量ベクトルが属すノードを探索し、ラベルを返す関数
        '''
        if self.threshold == None:
            return self.label
        
        if x[self.column] < self.threshold:
            return self.left.get_label(x)
        else:
            return self.right.get_label(x)            
        
    def predict(self, X):
        '''
        get_labelを特徴量行列の各行へ適用し、予測値ベクトルを得る関数
        '''
        pred = []
        N = X.shape[0]
        for i in range(N):
            x = X.loc[i]
            _y = node.get_label(x)
            pred.append(_y)
        return pred

if __name__ == "__main__":
    pass