"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def customer_bin(data, FEATURES_NUM):
    def custom_bin(x):
        if x == 0:
            return '[0]'
        if x > 0 and x <= 0.2:
            return '(0,0.2]'
        elif x > 0.2 and x <= 0.4 :
            return '(0.2,0.4]'
        elif x > 0.4 and x <= 0.6:
            return '(0.4,0.6]'
        elif x > 0.6 and x <= 0.8:
            return '(0.6,0.8]'
        elif x > 0.8 and x < 1:
            return '(0.8,1)'
        else:
            return '[1]'
 
    # Assumption -- heavy right skew, wide range (e.g. 0 all the way to 1000+)
    for feat in FEATURES_NUM:           
        # Check for low range e.g. percentage (fraction) feature between 0 and 1 or it's a "percentage" type
        if (data[feat].max()-data[feat].min() <= 1) or ('percent' in feat.split('_')):
            
            # For "open" related percentage, apply linear spacing, with 0 and 1 having their own bins
            if 'open' in feat.split('_'):
                bins = np.linspace(start=0, stop=1 ,num=6)
                bins = np.round(bins, 2)
                # Bin for zero (0)
                bins = np.insert(bins, 0, -0.00001)
                # Bin for one (1)
                bins = np.insert(bins, -1, 0.99999)
                # Out of bound values go into their respective bins
                if data[feat].max() > 1:
                    bins = np.append(bins, round(data[feat].max()+1))
                if data[feat].min() < 0:
                    bins = np.insert(bins, 0, round(data[feat].min()-1))
                data[f"{feat}_binned"] = pd.cut(data[feat], bins)
            
            # For "click" related percentage, apply geometric spacing, with 0 and 1 having their own bins
            elif 'click' in feat.split('_'):
                bins = np.geomspace(start=0.01, stop=1, num=6)
                bins = np.round(bins, 2)
                # Bin for zero (0)
                bins = np.insert(bins, 0, 0)
                bins = np.insert(bins, 0, -0.00001)
                # Bin for one (1)
                bins = np.insert(bins, -1, 0.99999)
                # Out of bound values go into their respective bins
                if data[feat].max() > 1:
                    bins = np.append(bins, round(data[feat].max()+1))
                if data[feat].min() < 0:
                    bins = np.insert(bins, 0, round(data[feat].min()-1))
                data[f"{feat}_binned"] = pd.cut(data[feat], bins)
                
            # For anything else, use custom binning
            else:
                data[f"{feat}_binned"] = data[feat].apply(custom_bin)
                data[f"{feat}_binned"] = data[f"{feat}_binned"].astype('category')
    
        # Put values in bins for heavily skewed wide range feature 0 to 100+
        # Zero (0) has its own bin
        else:
            
            # Define correction factor
            if data[feat].min() == 0:
                correction = 1
            else:
                correction = 2
            
            # Create uneven bins
            cnt, bins = np.histogram(np.log(data[feat]+correction), bins='doane')
            bins = np.unique(np.ceil(bins))
            bins = np.ceil(np.exp(bins))
            for i in range(correction+1):
                bins = np.insert(bins, 0, -i)
            bins = bins.astype('int')
            data[f"{feat}_binned"] = pd.cut(data[feat], bins)
    return data