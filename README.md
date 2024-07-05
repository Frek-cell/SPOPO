IMPORTANT: the "main" file actually contents the counterplots of the squeeaing degree. 

The code that provides the calculations themself is in the "bibl" file. The general scheme of the calculations is
  1. creating Omegas instance -- it describes the frequencies scale that is used in the current task (it is a list of frequencies)
  2. at each frequency point we calculate the spectraal density of the photocurrent and saving it in the Dot class instance
  3. the list of Dots, with its length equal to the length of Omegas, become united into the Line class instance
  4. together Line instance (or several of them) and Omegas instance form Data class instance

Data classs instance contains all the information about its Line instances, which can be shown in a form of pandas dataframe or plotted via matplotlib.
Unfortunately, counterplots of the squeezing degree can not be plotted via built-in methods, and therefore requires an additional code ("main" file) 
