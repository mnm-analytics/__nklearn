import pandas as pd

class GetOneHot:
    
    def __init__(self, train, test, tgt_cols, drop_first):
        self.train = train.copy()
        self.test = test.copy()
        self.TGT_COLS = tgt_cols
        self.offset = 1 if drop_first else 0
        self.dict_vals = {}
        for col in self.TGT_COLS:
            self.dict_vals[col] = set(self.train[col])
        
    def _gen_onehot(self, df):
        onehot_cols = []
        for k, val_set in self.dict_vals.items():
            val_list = list(val_set)[self.offset:]
            for v in val_list:
                ds = df[k].apply(lambda x: 1 if x == v else 0).rename(k + "_%s"%v)
                onehot_cols.append(ds)
        return pd.concat(onehot_cols, 1)
    
    def get_onehot(self):
        return self._gen_onehot(self.train), self._gen_onehot(self.test)
    
    def get_clns(self):
        train_oh, test_oh = self.get_onehot()
        train_rm, test_rm = self.train.drop(self.TGT_COLS, 1), self.test.drop(self.TGT_COLS, 1)
        
        return pd.concat([train_rm, train_oh],1), pd.concat([test_rm, test_oh],1)
    