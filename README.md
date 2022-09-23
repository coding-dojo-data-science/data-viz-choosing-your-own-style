# Choosing Your Own Visualization Style

- This notebook is designed to help select your own personal visualization aesthetic by using a combination of Matplotlib syles and Seaborn contexts.



```python
## Our standard import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs('images/', exist_ok=True)
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


    
![png](images/readme/output_8_0.png)
    


>- This is the figure we will be using to compare visualization aesthetics. This is what the figure looks like without any visualization customization.


```python
### temp setting context with sns.plotting_context
context = 'talk'
fontscale = .8

with sns.plotting_context(context=context,font_scale=fontscale):
    f,ax= test_plot(context)

```


    
![png](images/readme/output_10_0.png)
    


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


    
![png](images/readme/output_13_0.png)
    



    
![png](images/readme/output_13_1.png)
    



    
![png](images/readme/output_13_2.png)
    


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


    
![png](images/readme/output_17_0.png)
    



    
![png](images/readme/output_17_1.png)
    



    
![png](images/readme/output_17_2.png)
    



    
![png](images/readme/output_17_3.png)
    



    
![png](images/readme/output_17_4.png)
    



    
![png](images/readme/output_17_5.png)
    



    
![png](images/readme/output_17_6.png)
    



    
![png](images/readme/output_17_7.png)
    



    
![png](images/readme/output_17_8.png)
    



    
![png](images/readme/output_17_9.png)
    



    
![png](images/readme/output_17_10.png)
    



    
![png](images/readme/output_17_11.png)
    



    
![png](images/readme/output_17_12.png)
    



    
![png](images/readme/output_17_13.png)
    



    
![png](images/readme/output_17_14.png)
    



    
![png](images/readme/output_17_15.png)
    



    
![png](images/readme/output_17_16.png)
    



    
![png](images/readme/output_17_17.png)
    



    
![png](images/readme/output_17_18.png)
    



    
![png](images/readme/output_17_19.png)
    



    
![png](images/readme/output_17_20.png)
    



    
![png](images/readme/output_17_21.png)
    



    
![png](images/readme/output_17_22.png)
    



    
![png](images/readme/output_17_23.png)
    



    
![png](images/readme/output_17_24.png)
    



    
![png](images/readme/output_17_25.png)
    


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


    
![png](images/readme/output_20_0.png)
    



    
![png](images/readme/output_20_1.png)
    



    
![png](images/readme/output_20_2.png)
    



    
![png](images/readme/output_20_3.png)
    


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


    
![png](images/readme/output_24_0.png)
    


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


    
![png](images/readme/output_26_0.png)
    



    
![png](images/readme/output_26_1.png)
    



    
![png](images/readme/output_26_2.png)
    



    
![png](images/readme/output_26_3.png)
    



    
![png](images/readme/output_26_4.png)
    


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




    
![png](images/readme/output_32_1.png)
    


# BOOKMARK: HERE TO END NEEDS UPDATING

## Selecting Parameters for `fig.savefig`

There are several parameters in plt.rcParams that are used to determine the quality and aesthetics of saved figures. 

- Three of the most important settings to check are:
    - `savefig.dpi`: the figure quality - dpi (displayed-pixels per inch)
        - default value = "figure" (whatever the figure was using).
        - preferred value= 300
            - by setting this to 300, we are producing high resolution images that could be used in professional printed publications.
            
    - `savefig.bbox`:  how much extra spacing is around the figure's Axis (the bounding box).
        - default value = None. <br>- Uses the default bounding box of the figure. 
        - preferred value="tight".<br> - It will determine a new bounding box that will be big enough to include all text labels in the exported image. 
        
    - `savefig.transparent`: if the figure will have a transparent background. 
        - default value = `False.
        - preferred value = `False`<br> but this parameter doesn't always work to prevent a transparent backround.
        - There are still situations where if the image is displayed on a page with a dark background (like when using GitHub's dark theme), then the text labels for the plot may be obscured.
    - `savefig.facecolor`: the facecolor of the figure (the background of the entire figure - NOT the same as the color of the plotting area inside of the axes)
        - default value = 'auto'
        - preferred value = depends on your matplotlib style. You may want: 'white', 'black', or 'gray' depending on what looks best with your selected visualization settings.



```python
# whats the default value?
print(plt.rcParams['savefig.dpi'])
```

    figure



```python
# whats the default value?
print(plt.rcParams['savefig.bbox'])
```

    None



```python
# whats the default value?
print(plt.rcParams['savefig.transparent'])
```

    False



```python
# whats the default value?
print(plt.rcParams['savefig.facecolor'])
```

    auto


## Testing fig.savefig with Default Image Settings


```python
# creating a fig using the LongFjob feature (since it has long labels)
fig, ax = test_plot(x1='LongFjob',rotate=True)
```


    
![png](images/readme/output_41_0.png)
    



```python
# saving the figure wiht the default params
fig.savefig('images/test_plot_defaults.png')
```

- Now that we have saved our image, the cell below is using HTML and CSS to create an area with a black background and then it inserts our image inside of the black area. 
- We need to confirm that we can see the text labels in our visualization.

### Testing Using Saved Image Against a Dark Background

<div style='background-color:black'>
    <p style='color:white'> Testing Inserting Image in a Dark Background</p>

 <img src="images/test_plot_defaults.png">
    <p style='color:white'> Testing Inserting Image in a Dark Background </p></div>


- Notice in the image exporting using the default values:
    - [ ] The text labels are cut off!
    - [x] The background is not transparent.
    - [x] The image doesn't look very blurry.

### Updating savefig params


```python
plt.rcParams['savefig.dpi'] = 300


plt.rcParams['savefig.bbox'] = 'tight'


plt.rcParams['savefig.transparent'] = False


plt.rcParams['savefig.facecolor'] = 'white'
```


```python
# saving the figure wiht the default params
fig, ax = test_plot(x1='LongFjob',rotate=True)
fig.savefig('images/test_plot-new_params.png')
```


    
![png](images/readme/output_48_0.png)
    



```python

```

### Testing Updated Params

#### Testing Using Saved Image Against a Dark Background

<div style='background-color:black'>
    <p style='color:white'> Testing Inserting Image in a Dark Background </p>

 <img src="images/test_plot-new_params.png">
    <p style='color:white'> Testing Inserting Image in a Dark Background </p>
</div>



```python
plt.imshow(plt.imread('images/test_plot-new_params.png'))
```




    <matplotlib.image.AxesImage at 0x14f7875e0>




    
![png](images/readme/output_52_1.png)
    


## Export Final Choices

- Paste your final plt.rcParams for savefig in the f-string below. 

- Run the cell to see the final code for setting the visualization aesthetics. 

- If everything looks correct, run the following cell to save these settings as a new py file called `my_style.py`.




```python
final_favs = f"""
import matplotlib.pyplot as plt
import seaborn as sns

fav_style = {fav_style}
fav_context  ={fav_context}
plt.style.use(fav_style)
sns.set_context(**fav_context)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.facecolor'] = 'white'
"""

print(final_favs)
```

    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fav_style = ('ggplot', 'tableau-colorblind10')
    fav_context  ={'context': 'notebook', 'font_scale': 1.2}
    plt.style.use(fav_style)
    sns.set_context(**fav_context)
    
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.transparent'] = False
    plt.rcParams['savefig.facecolor'] = 'white'
    



```python
## Saving the paramets to my_style.py
with open('my_style.py', 'w+') as f:
    f.write(final_favs)
    
with open('my_style.py', 'r') as f:
    print(f.read())


```

    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fav_style = ('ggplot', 'tableau-colorblind10')
    fav_context  ={'context': 'notebook', 'font_scale': 1.2}
    plt.style.use(fav_style)
    sns.set_context(**fav_context)
    
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.transparent'] = False
    plt.rcParams['savefig.facecolor'] = 'white'
    


## RESTART KERNEL AND RUN ONLY CELLS BELOW TO TEST

- Final Step: Restart you Kernel to reset your notebook's settings. 
- DO NOT RUN ANY CODE ABOVE HERE, just run the following cell to use your saved settings.
- Then run the final cells to show the final result.

Any notebook that is stored in the same folder as your my_style.py file can run the following to use the settings:

```python
from my_style import *
```


```python
from my_style import *
```


```python
## TESTING VIZ PARAMS 
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
fig.savefig('images/final_test_plot.png')
```


    
![png](images/readme/output_60_0.png)
    



<div style='background-color:black'>
    <p style='color:white'> Testing Inserting Image in a Dark Background </p>

 <img src="images/final_test_plot.png">
    <p style='color:white'> Testing Inserting Image in a Dark Background </p>
</div>



```python
plt.imshow(plt.imread('images/final_test_plot.png'))
```




    <matplotlib.image.AxesImage at 0x1563fd880>




    
![png](images/readme/output_62_1.png)
    

