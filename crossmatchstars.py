# Compares 2 lists of stars and puts duplicates into a second column thus showing the number of unique stars between the 2

# file1 is larger set. file2 is smaller set.
#maxsep is the max separation 2 stars can be and still be considered a match.
#reg is the region the 2 catalogs are in.
#paper1 and paper 2 are the source authors or whatever you want in relation to file1 and file2

# column names required in each of the df for this to work are: 
#"_RAJ2000","_DEJ2000","SpT","Reg","source"


def crossmatch(file1, file2, maxsep=0.5, reg='region', paper1='paper1', paper2="paper2"):
    ra1 = file1['_RAJ2000']
    dec1 = file1['_DEJ2000']
    ra2 = file2['_RAJ2000']
    dec2 = file2['_DEJ2000']
    max_sep= maxsep*u.arcsec
    
    #convert to skycoords
    f1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree) #list with more stars
    f2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree) #list with less stars
    
    #comparing file2 to file 1 and outputting the indicies for f1 and f2 which correspond
    #to duplicates
    idf2, idf1, d2d, d3d = f1.search_around_sky(f2, seplimit = max_sep)

    #Before combining the data, change the column names for the smaller set or else it wont 
    #concat with the full list and give an error. So here f2 is smaller than f1, and I want 
    #the results to match side by side so that each set of matches only counts as 1 index. 
    #When I append the list of duplicates to the list of unique sources, and  it sees repeated
    #columns in the duplicates df, it gets confused which column names to match to

    #Reset the index. Index needs to reset for the output of astropy to flag the correct stars. 
    #Essentially you can have situations where the df index starts at 60 instead of 1, the 
    #output of idx1 of star 5 wont return anything.
    xmatchf2 = file2.iloc[idf2].reset_index(drop=True).rename(columns={"SpT": f"SpT_{paper2}","index": f"index_{paper2}", "_RAJ2000": f"_RAJ2000_{paper2}", "_DEJ2000": f"_DEJ2000_{paper2}", "Reg": f"Reg_{paper2}", "source": f"source_{paper2}"})
    xmatchf1 = file1.iloc[idf1].reset_index(drop=True)
    #combine the duplicates to be side by side for comparison purposes
    duplicates = pd.concat([xmatchf1, xmatchf2], axis=1)

    #create a list of stars unique to each df by dropping the matches
    f2unique = file2.drop(idf2)
    f1unique = file1.reset_index(drop=True).drop(idf1)

    unique = pd.concat([f2unique,f1unique])

    #return a df of the full list of stars, where repeats are side by side
    return pd.concat([unique,duplicates])
