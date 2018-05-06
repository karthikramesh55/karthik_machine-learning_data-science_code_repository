import codecademylib
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency

#Importing the Species Information dataframe
species = pd.read_csv("species_info.csv", sep = ',')
print species.head(5)

#Different species in the dataframe
species_count = species['scientific_name'].nunique()
print "\n"
print "Unique Species Count: %d" %(species_count)

#Different categories of animals in the dataframe
species_type = species['category'].unique()
print "\n"
print "Unique Animal Categories: ", species_type

#Different conservation statuses
conservation_statuses = species['conservation_status'].unique()
print "\n"
print "Unique Conservation Statuses: ", conservation_statuses

#Grouping animal species based on their conservation statuses
conservation_counts = species.groupby('conservation_status').scientific_name.nunique().reset_index()
print "\n"
print "Segmentation of animal species based on their conservation status" + "\n\n", conservation_counts

#Renaming the conservation status NaN with 'No Intervention'
species.fillna('No Intervention', inplace = True)

#Grouping animal species based on their conservation statuses after renaming the NaN category
conservation_counts_fixed = species.groupby('conservation_status').scientific_name.nunique().reset_index()
print "\n"
print "Segmentation of animal species based on their conservation status" + "\n\n", conservation_counts_fixed


#Sorting the grouped dataframe by the alphabetical order of animal species
protection_counts = species.groupby('conservation_status').scientific_name.nunique().reset_index().sort_values(by='scientific_name')

#Plotting the Conservation Status by species
plt.figure(figsize=(10, 4))
ax = plt.subplot()
plt.bar(range(len(protection_counts)),protection_counts.scientific_name.values)
ax.set_xticks(range(len(protection_counts)))
ax.set_xticklabels(protection_counts.conservation_status.values)
plt.ylabel('Number of Species')
plt.title('Conservation Status by Species')
plt.show()

#Finding whether various species are protected or not
species['is_protected'] = species.conservation_status != 'No Intervention'
category_counts = species.groupby(['category', 'is_protected']).scientific_name.nunique().reset_index()
print category_counts.head()
category_pivot = category_counts.pivot(columns='is_protected', index='category', values='scientific_name').reset_index()
print category_pivot

#Percentage of protected species by category
category_pivot.columns = ['category', 'not_protected', 'protected']   #Renaming the 'False' and 'True' columns (signifying protection) as 'Not Protected' and 'Protected' respectively
category_pivot['percent_protected'] = 100 * ( category_pivot.protected / (category_pivot.protected + category_pivot.not_protected))
print category_pivot

#Contingency test to check if Mammals are more endangered than Birds
#Null Hypothesis: There is no significant difference between the two endangerment likelihoods of the 2 categories. It would have happened by chance.
#Technique used: Chi Square Contingency Test

contingency = [[30, 146], [75, 143]]
tstat, p_val, dof, expected_value = chi2_contingency(contingency)
print "The p-value from the contingency test is: %0.2f", %(p_val)
if p_val < 0.05:
	print "There is significant difference. Null hypothesis is false"
else:
	print "There is no significant difference. Null hypothesis is true"

#P-Value: 0.687594809666. No significant difference.


#Contingency test to check if Mammals are more endangered than Reptiles
#Null Hypothesis: There is no significant difference between the two endangerment likelihoods of the 2 categories. It would have happened by chance.
#Technique used: Chi Square Contingency Test

contingency_mammal_reptile = [[30, 146], [5, 73]]
tstat_2, pval_mammal_reptile, dof_2, expected_value_2 = chi2_contingency(contingency_mammal_reptile)
if pval_mammal_reptile < 0.05:
	print "There is significant difference. Null hypothesis is false"
else:
	print "There is no significant difference. Null hypothesis is true"

#P-Value: 0.0383555902297. There's significant difference. Certain species are more endangered than others

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#National Park Service sends over another dataset with observations made on various species across various national parks in a week

observations = pd.read_csv('observations.csv')
print observations.head(5)

#Tracking sheep in the Species dataframe
species['is_sheep'] = species.common_names.apply(lambda x: 'Sheep' in x)
species_is_sheep = species[species.is_sheep]
print species_is_sheep

#We find many vascular plants having the keyword 'Sheep' in their names. So we separate the actual sheep species by using the Mammal category as a reference
sheep_species = species[(species.is_sheep) & (species.category == 'Mammal')]
print sheep_species

#Merging the Sheep Species dataframe with the Observations dataframe
sheep_observations = observations.merge(sheep_species)
print sheep_observations.head()

#Sheep sighting observations in each national park
obs_by_park = sheep_observations.groupby('park_name').observations.sum().reset_index()
print obs_by_park

#Plotting sheep sightings
plt.figure(figsize=(16,4))
ax = plt.subplot()
plt.bar(range(len(obs_by_park)), obs_by_park.observations.values)
ax.set_xticks(range(len(obs_by_park)))
ax.set_xticklabels(obs_by_park.park_name.values)
plt.ylabel('Number of Observations')
plt.title('Observations of Sheep per Week')
plt.show()

#From the above plot, we observe that Yellowstone National Park has the highest sheep sightings, and Great Smoky Mountain National Park has the least sheep sightings

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Sample size determination for detecting atleast 5% of feverish sheep

#Baseline rate: 15% or 0.15
#Statistical significance = 90%
#Minimum detectable effect = 100 * x / Baseline rate = 100 * 5 / 15 = 33% or 0.33

sample_size_per_variant = 890

#Number of weeks the scientists must spend at Yellowstone National Park and Bryce National Park to observe enough sheep
yellowstone_weeks_observing = sample_size_per_variant/507.   #507 is the number of observations per week at Yellowstone National Park
print yellowstone_weeks_observing                            #1 week

bryce_weeks_observing = sample_size_per_variant/250.         #250 is the number of observations per week at Bryce National Park
print bryce_weeks_observing                                  #3 weeks

#Inferences: 
"""Given a baseline of 15% occurrence of foot and mouth disease in sheep at Bryce National Park,you found that if the scientists wanted to be sure that a >5% drop in observed 
cases of foot and mouth disease in the sheep at Yellowstone was significant they would have to observe at least 510 sheep.Then, using the observation data analyzed earlier, we 
found that this would take approximately one week of observing in Yellowstone to see that many sheep, or approximately three weeks in Bryce to see that many sheep."""






























