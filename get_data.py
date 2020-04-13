#!/usr/bin/python
# -*- Coding: utf-8 -*-

import datetime as dt
import pandas as pd
import numpy as np

def get_day_week_list():
    # 内閣府発表の祝日・振替休日リスト
    tgt = "http://www8.cao.go.jp/chosei/shukujitsu/syukujitsu_kyujitsu.csv"
    holiday = pd.read_csv(tgt, encoding="sjis")
    
    # clns
    holiday.columns = ["date", "name"]
    holiday["date"] = holiday.date.apply(lambda s: dt.datetime.strptime(s, '%Y-%m-%d'))
    holiday["is_holiday"] = 1
    
    # get days
    d_max = holiday.date.max()
    d_min = holiday.date.min()
    diff_max_mini = (d_max - d_min).days
    print(diff_max_mini, " days: from ", d_min, " to ", d_max)
    
    d = [holiday.date.min() + dt.timedelta(int(i)) for i in np.arange(diff_max_mini)]
    d = pd.DataFrame({"date":d})
    d["is_weekend"] = d["date"].apply(dt.datetime.weekday).apply(lambda wd: 1 if wd >= 5 else 0)
    
    # merge
    df = pd.merge(d, holiday, "left", on="date")
    tgt_col = ["date", "is_weekend", "is_holiday"]
    days = df[tgt_col].fillna(0)
    
    # gen weeks
    weeks = days.copy()
    weeks["week"] = weeks["date"] - pd.to_timedelta(weeks["date"].apply(dt.datetime.weekday), "d")
    weeks = weeks.groupby("week")["is_holiday"].max()
    weeks = pd.DataFrame(weeks)
    
    return days, weeks

if __name__ == "__main__":
    pass