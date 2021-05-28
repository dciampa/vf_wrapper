import numpy as np
# Find all line numbers in the file that have the target name,
# which act like mini headers
lnum=[] # Create a blank list for the line numbers of the header rows.
transition_list=[] # Blank list for the ion and wavelength from header rows in the spectral file
with open(data_directory+file) as myFile:
    # Read all of each line in the file and store the information as line.
    # For each line read in, num will grow by 1.
    for num, line in enumerate(myFile, 1):
        # Search for the target name in the read in line. If it matches, save that line number.
        if target in line:
            lnum=lnum+[num] # Concatinate the lnum list with the line number with target name in it.
            # Split the text where ever there is a dash, plus, or space
            tmp=re.split("[-+ ]", line)
            # The split on tmp[3] is to remove any periods from the line transition strings
            transition=[tmp[2]+'_'+tmp[3].split(".")[0]] # square brackets are needed to make this a list
            transition_list=transition_list+transition # ion+line list of strings
    lnum=lnum+[num] # To grab the last line in the data file
