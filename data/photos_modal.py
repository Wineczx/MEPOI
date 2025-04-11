import json

# Read the original JSON lines file and collect reviews by business_id
reviews_by_business = {}
input_filename = '/data/yelp/PA/PA_photostoreview.json'  # Change this to your input filename
output_filename = '/data/yelp/PA/image2review.json'  # Change this to your desired output filename

with open(input_filename, 'r') as input_file:
    for line in input_file:
        review_data = json.loads(line)
        business_id = review_data['business_id']
        review_text = review_data['review']
        # Initialize the list of reviews for new business_id or append to existing list
        if business_id not in reviews_by_business:
            reviews_by_business[business_id] = []
        reviews_by_business[business_id].append(review_text)

# Write the aggregated reviews to a new JSON file
with open(output_filename, 'w') as output_file:
    # Write each business and its associated reviews as a JSON object
    for business_id, reviews in reviews_by_business.items():
        business_reviews = {business_id: reviews}
        json.dump(business_reviews, output_file)
        output_file.write('\n')  # Ensure each JSON object is on a new line
