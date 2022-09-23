# Choosing Your Own Visualization Style

- This notebook is designed to help select your own personal visualization aesthetic by using a combination of Matplotlib syles and Seaborn contexts.



```python
## Our standard import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
## Load in the student performance - math dataset & display the head and info
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS6xDKNpWkBBdhZSqepy48bXo55QnRv1Xy6tXTKYzZLMPjZozMfYhHQjAcC8uj9hQ/pub?output=xlsx"

df = pd.read_excel(url,sheet_name='student-mat')

## making a feature with longer names for testing savefig
df['LongFjob'] = df['Fjob'] +'__' + df['Fjob'] + '__' + df['Fjob']

df.info()
df.head(3)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 395 entries, 0 to 394
    Data columns (total 34 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   school      395 non-null    object 
     1   sex         395 non-null    object 
     2   age         395 non-null    float64
     3   address     395 non-null    object 
     4   famsize     395 non-null    object 
     5   Pstatus     395 non-null    object 
     6   Medu        395 non-null    float64
     7   Fedu        395 non-null    float64
     8   Mjob        395 non-null    object 
     9   Fjob        395 non-null    object 
     10  reason      395 non-null    object 
     11  guardian    395 non-null    object 
     12  traveltime  395 non-null    float64
     13  studytime   395 non-null    float64
     14  failures    395 non-null    float64
     15  schoolsup   395 non-null    object 
     16  famsup      395 non-null    object 
     17  paid        395 non-null    object 
     18  activities  395 non-null    object 
     19  nursery     395 non-null    object 
     20  higher      395 non-null    object 
     21  internet    395 non-null    object 
     22  romantic    395 non-null    object 
     23  famrel      395 non-null    float64
     24  freetime    395 non-null    float64
     25  goout       395 non-null    float64
     26  Dalc        395 non-null    float64
     27  Walc        395 non-null    float64
     28  health      395 non-null    float64
     29  absences    395 non-null    float64
     30  G1          395 non-null    float64
     31  G2          395 non-null    float64
     32  G3          395 non-null    float64
     33  LongFjob    395 non-null    object 
    dtypes: float64(16), object(18)
    memory usage: 105.0+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
      <th>LongFjob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GP</td>
      <td>F</td>
      <td>18.0</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>at_home</td>
      <td>teacher</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>teacher__teacher__teacher</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP</td>
      <td>F</td>
      <td>17.0</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>other__other__other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GP</td>
      <td>F</td>
      <td>15.0</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>other__other__other</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 34 columns</p>
</div>



### Check installed plotting packages


```python
import matplotlib
matplotlib.__version__
```




    '3.4.3'




```python
import seaborn as sns
sns.__version__
```




    '0.11.2'



### Function for quick `test_plot` 


```python
## TESTING VIZ PARAMS [temp]

def test_plot(title='Testing Contexts',x1='Fjob',rotate=False):
    fig,axes = plt.subplots(ncols=2,figsize=(12,4))
    sns.barplot(data=df,x=x1,y='G3',ax=axes[0])
    sns.regplot(data=df,x='G1',y='G3',ax=axes[1])
    fig.suptitle(title)
    
    if rotate:
        axes[0].set_xticklabels(axes[0].get_xticklabels(),
                            rotation=45, ha='right')
    else:
        fig.tight_layout()        
    
    return fig,axes

test_plot();
```


    
![png](images/output_8_0.png)
    


>- This is the figure we will be using to compare visualization aesthetics. This is what the figure looks like without any visualization customization.


```python
### temp setting context with sns.plotting_context
context = 'talk'
fontscale = .8

with sns.plotting_context(context=context,font_scale=fontscale):
    f,ax= test_plot(context)

```


    
![png](images/output_10_0.png)
    


## Testing Options for `sns.set_context`
Various seaborn contexts + font scales

- To test different combinations of seaborn contexts + font_scale options, create a list of tuples called `text_contexts` where each tuple contains:
    1. a seaborn context  (e.g. talk,notebook,poster, paper)
    2. a value for font_scale (e.g. 0.8,1.0,1.1)
    
    
```python
## Example list of tuples with (context, fontscale)
test_contexts = [('talk',.8), ('notebook',1.1), ('paper',1.1)]
```


```python
# YOUR OPTIONS HERE
## context, fontscale
test_contexts = [('talk',.8), ('notebook',1.1), ('paper',1.1)]



for (context,font_scale) in test_contexts:
    
    with sns.plotting_context(context=context,
                              font_scale=font_scale):
        f,ax= test_plot(title=f"{context}: (fontscale={font_scale})")
```


    
![png](images/output_13_0.png)
    



    
![png](images/output_13_1.png)
    



    
![png](images/output_13_2.png)
    


## Testing Matplotlib Styles (for `plt.style.use`)

- Let's preview all of the available named styles in Matplotlib and then decide which ones we want to try combining into 1 final style.


```python
# all styles available
plt.style.available
```




    ['Solarize_Light2',
     '_classic_test_patch',
     'bmh',
     'classic',
     'dark_background',
     'fast',
     'fivethirtyeight',
     'ggplot',
     'grayscale',
     'seaborn',
     'seaborn-bright',
     'seaborn-colorblind',
     'seaborn-dark',
     'seaborn-dark-palette',
     'seaborn-darkgrid',
     'seaborn-deep',
     'seaborn-muted',
     'seaborn-notebook',
     'seaborn-paper',
     'seaborn-pastel',
     'seaborn-poster',
     'seaborn-talk',
     'seaborn-ticks',
     'seaborn-white',
     'seaborn-whitegrid',
     'tableau-colorblind10']




```python
## Loop to create plot using style temporarily 
for style in plt.style.available:
    with plt.style.context(style):
        f,ax= test_plot(title=style)
        plt.show()
        del f
```


    
![png](images/output_17_0.png)
    



    
![png](images/output_17_1.png)
    



    
![png](images/output_17_2.png)
    



    
![png](images/output_17_3.png)
    



    
![png](images/output_17_4.png)
    



    
![png](images/output_17_5.png)
    



    
![png](images/output_17_6.png)
    



    
![png](images/output_17_7.png)
    



    
![png](images/output_17_8.png)
    



    
![png](images/output_17_9.png)
    



    
![png](images/output_17_10.png)
    



    
![png](images/output_17_11.png)
    



    
![png](images/output_17_12.png)
    



    
![png](images/output_17_13.png)
    



    
![png](images/output_17_14.png)
    



    
![png](images/output_17_15.png)
    



    
![png](images/output_17_16.png)
    



    
![png](images/output_17_17.png)
    



    
![png](images/output_17_18.png)
    



    
![png](images/output_17_19.png)
    



    
![png](images/output_17_20.png)
    



    
![png](images/output_17_21.png)
    



    
![png](images/output_17_22.png)
    



    
![png](images/output_17_23.png)
    



    
![png](images/output_17_24.png)
    



    
![png](images/output_17_25.png)
    


### Testing Combinations of Styles 

- Based on the styles above, create a list called `test_combined` that contains a list of tuples with each tuple containing the names of multiple styles to combine. 
    - Note: the order does matter, so for any pair of styles, you should try both orders.
    
```python
# list of tuples with (style1, style2) (or more!)
test_combined = [('tableau-colorblind10','fivethirtyeight'),
                ('fivethirtyeight','tableau-colorblind10')]
```


```python
# YOUR OPTIONS HERE
# list of tuples with (style1, style2) (or more!)
test_combined = [('tableau-colorblind10','fivethirtyeight'),
                ('fivethirtyeight','tableau-colorblind10'),
                ('ggplot','tableau-colorblind10'),
                ('tableau-colorblind10','ggplot',)]



## Loop to test out every style combination
for style in test_combined:
    with plt.style.context(style):
        f,ax= test_plot(title=f"{style[0]},{style[1]}")
        plt.show()
        del f
```


    
![png](images/output_20_0.png)
    



    
![png](images/output_20_1.png)
    



    
![png](images/output_20_2.png)
    



    
![png](images/output_20_3.png)
    


# Selecting Final Fav Style & Re-test contexts

### Select Fav Style

- Based on the outputs above, select your final favorite matplotlib style (or tuple of styles) and save them as `fav_style`


```python
## YOUR FAV_STYLE HERE
fav_style = ('ggplot','tableau-colorblind10')


## Previewing fav_style temporarily
with plt.style.context(fav_style):
    test_plot(title=str(fav_style));
```


    
![png](images/output_24_0.png)
    


### Test Options for `sns.set_context` with Fav Style


```python
# YOUR OPTIONS HERE
## context, fontscale
test_contexts = [('talk',.8), ('notebook',1.1), ('paper',1.1),
                ('notebook',1.2)]

with plt.style.context(fav_style):
    test_plot(title=str(fav_style));
    for (context,font_scale) in test_contexts:

        with sns.plotting_context(context=context,
                                  font_scale=font_scale):
            f,ax= test_plot(title=f"{context}: (fontscale={font_scale})")
```


    
![png](images/output_26_0.png)
    



    
![png](images/output_26_1.png)
    



    
![png](images/output_26_2.png)
    



    
![png](images/output_26_3.png)
    



    
![png](images/output_26_4.png)
    


### Saving the Final `fav_context`

- Save the final values for your `context` and `font_sale` into a dict called `fav_context`.

```python
# example fav_context
fav_context  ={'context':'notebook', 'font_scale':1.2}
```


```python
fav_context  ={'context':'notebook', 'font_scale':1.2}
```

### Final Style Choices


```python
## Print fav_style and fav_context to confirm
print(fav_style)
print(fav_context)
```

    ('ggplot', 'tableau-colorblind10')
    {'context': 'notebook', 'font_scale': 1.2}



```python
## SET THE STYLE AND SET THE CONTEXT
plt.style.use(fav_style)
sns.set_context(**fav_context)

## Generate Final Test Plot
test_plot()
```




    (<Figure size 1200x400 with 2 Axes>,
     array([<AxesSubplot:xlabel='Fjob', ylabel='G3'>,
            <AxesSubplot:xlabel='G1', ylabel='G3'>], dtype=object))




    
![png](images/output_32_1.png)
    


# BOOKMARK: HERE TO END NEEDS UPDATING

## APPENDIX

### Testing fig.savefig Image Settings


```python
fig, ax = test_plot(x1='LongFjob',rotate=True)
```


    
![png](images/output_36_0.png)
    



```python
fig.savefig('test_plot_trans.png')#,transparent=True)
```

- Testing Image: Transparent=True


<div style='background-color:black'> <img src="test_plot_trans.png">Testing</div>



```python
fig.savefig('test_plot_not_t.png',transparent=False)
```

- Testing Image: Transparent=False
<div style='background-color:black'> <img src="test_plot_not_t.png">Testing</div>


### Using Savefig rcParams


```python
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.bbox'] = 'tight'
```


```python
##final test
fig.savefig('test_plot_final.png')
```


```python
plt.imshow(plt.imread('test_plot_final.png'))
```




    <matplotlib.image.AxesImage at 0x130b6a970>




    
![png](images/output_44_1.png)
    


- Testing Final Image using rcParams
<div style='background-color:black'> <img src="test_plot_final.png">Testing</div>


## Export Final Choices

- Save the final choices as final use commands in the lp_styles.py file.


```python
final_favs = """import matplotlib.pyplot as plt
import seaborn as sns
fav_style = ('ggplot','tableau-colorblind10')
fav_context  ={'context':'notebook', 'font_scale':1.2}
plt.style.use(fav_style)
sns.set_context(**fav_context)
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.bbox'] = 'tight'"""
```


```python

with open('lp_styles.py', 'w+') as f:
    f.write(final_favs)
    
with open('lp_styles.py', 'r') as f:
    print(f.read())

    

```

    import matplotlib.pyplot as plt
    import seaborn as sns
    fav_style = ('ggplot','tableau-colorblind10')
    fav_context  ={'context':'notebook', 'font_scale':1.2}
    plt.style.use(fav_style)
    sns.set_context(**fav_context)
    plt.rcParams['savefig.transparent'] = False
    plt.rcParams['savefig.bbox'] = 'tight'


## RESTART KERNEL AND RUN ONLY CELLS BELOW TO TEST


```python
from lp_styles import *
```


```python
## TESTING VIZ PARAMS [temp]
import pandas as pd
## Load in the student performance - math dataset & display the head and info
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS6xDKNpWkBBdhZSqepy48bXo55QnRv1Xy6tXTKYzZLMPjZozMfYhHQjAcC8uj9hQ/pub?output=xlsx"

df = pd.read_excel(url,sheet_name='student-mat')
## testing longer names
df['LongFjob'] = df['Fjob'] +'__' + df['Fjob'] + '__' + df['Fjob']


def test_plot(title='Testing Contexts',x1='Fjob',rotate=False):
    fig,axes = plt.subplots(ncols=2,figsize=(12,4))
    sns.barplot(data=df,x=x1,y='G3',ax=axes[0])
    sns.regplot(data=df,x='G1',y='G3',ax=axes[1])
    fig.suptitle(title)
    
    if rotate:
        axes[0].set_xticklabels(axes[0].get_xticklabels(),
                            rotation=45, ha='right')
    else:
        fig.tight_layout()   
    return fig, axes
    

fig, axes = test_plot(x1='LongFjob',rotate=True)
fig
```




    
![png](images/output_52_0.png)
    




    
![png](images/output_52_1.png)
    



```python
fig.savefig('test_plot_t.png')#,transparent=True)
```

- Testing Image: Transparent=True
<div style='background-color:black'> <img src="test_plot_t.png">Testing</div>



```python
fig.savefig('test_plot_not_t.png',transparent=False)
```

- Testing Image: Transparent=False
<div style='background-color:black'> <img src="test_plot_not_t.png">Testing</div>



```python
fig.savefig('test_plot_final.png')
```


```python
plt.imshow(plt.imread('test_plot_final.png'))
```




    <matplotlib.image.AxesImage at 0x127c52160>




    
![png](images/output_58_1.png)
    



```python

```
