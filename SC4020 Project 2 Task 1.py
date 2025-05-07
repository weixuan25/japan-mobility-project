import csv
import pandas as pd # type: ignore
from mlxtend.frequent_patterns import apriori # type: ignore
from mlxtend.frequent_patterns import association_rules # type: ignore
from mlxtend.preprocessing import TransactionEncoder # type: ignore


def read_csv(file1):
	POIdata= [ ]

	with open(file1, 'r') as file:            # opens and read the file
		reading_cityA = csv.reader(file)
		next(reading_cityA)                                # it will skip the header row
	
		for i, row in enumerate(reading_cityA):
			if i < 30:                                      # to read only the first 30 days
				POIdata.append(row)
			else: 
				break 
	return POIdata
			
def read_categories(file1):	    # to read categories of POI
	CategoryOfPOI = [ ] 				    # to store the categories
		
	with open(file1, 'r') as file:
		reading_category= csv.reader(file)
		for row in reading_category:
			CategoryOfPOI.append(row [0])

	return CategoryOfPOI

POIdata = read_csv('POIdata_cityA.csv')
CategoryOfPOI = read_categories('POI_datacategories.csv')

def process_data(POIdata, CategoryOfPOI):
	processedData = [ ] 
		
	for row in POIdata:
		if len(row) < 4:
			print(f"Skipping invalid row:{row}")
			continue  # Skip rows that don't have 4 elements (x, y, category_index, POI_count)

		x, y, category_index, POI_count = row

		# Ensure category_index is an integer
		try:
			category_index = int(category_index)  # Convert to integer if it's a string
		except ValueError:
			print(f"Invalid category_index: {category_index}, skipping row.")
			continue  # Skip this row if conversion fails

		# Ensure POI_count is an integer
		try:
			POI_count = int(POI_count)  # Convert POI_count to an integer
		except ValueError:
			print(f"Invalid POI_count: {POI_count}, skipping row.")
			continue  # Skip this row if conversion fails
	
		if category_index < len(CategoryOfPOI):	
			Name_of_category = CategoryOfPOI[category_index]
		else:
			Name_of_category = "Unknown"

		processedData.append({ 
			'x': x,
			'y': y,
			'category': Name_of_category,
			'POI_count': POI_count
		
			})

	return processedData
	
POIdata = read_csv('POIdata_cityA.csv')	
CategoryOfPOI = read_categories('POI_datacategories.csv')
	
processedData = process_data(POIdata, CategoryOfPOI)

df = pd.DataFrame(processedData)

Basket = df.groupby([ 'x', 'y' ]).apply(
	lambda x: x.apply(
		lambda row: [row['category']] * int(row['POI_count']), axis=1
	).sum()
).reset_index()      # grouping of grid coordinates according to their columns

Baskets_in_columns = ['x', 'y', 'Baskets']                 # renaming of columns
Basket.columns = ['x', 'y', 'Baskets']

transactions = Basket['Baskets'].tolist()
transactions = [set(transaction) for transaction in transactions]

if not transactions:
    print("No transactions found!")

print("Transactions (Baskets):")
print(transactions)

trans_encoder = TransactionEncoder()
trans_encoder_ary = trans_encoder.fit_transform(transactions)         # to transform list of baskets into a one-hot encoded format

df_trans_encoder = pd.DataFrame(trans_encoder_ary, columns = trans_encoder.columns_)   # convert to Pandas DataFrame

frequent_itemsets = apriori(df_trans_encoder, min_support = 0.1, use_colnames = True)

association_rules_df = association_rules(frequent_itemsets, metric = "lift", min_threshold = 0.5, num_itemsets=10)

print("Frequent Itemsets:")
	  
print(frequent_itemsets)

print("Association Rules:")
	  
print(association_rules_df)