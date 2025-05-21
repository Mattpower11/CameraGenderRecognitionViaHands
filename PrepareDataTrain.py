import pandas as pd
import numpy as np

def prepare_data_train(csv_path:str, num_exp: int, num_train: int):
    # Load the data from csv metadata file
    df = pd.read_csv(csv_path)

    # Create a data structure to store the images' name and the corresponding label
    data_structure = {}
    
    gender = ['male',  'female']

    print("Preparing Data For Training\n")

    # Populate the data structure
    for indExp in range(num_exp):
        print(f"\tExp {indExp}")
        data_structure[indExp] = {
                                    "labels": [],
                                    "images": []
                                }      
        df['check'] = False

        for gend in gender:
            # Extract the person id without accessories
            person_id_list = df.loc[(df['gender'] == gend), 'id'].unique()
            
            for _ in range(num_train):
                for i in range(0, len(person_id_list)):
                    # Extract a person id
                    person_id = np.random.choice(person_id_list)

                    '''
                        Exclude people who no longer have palm and back images to extract
                    '''
                    if (len(df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar'))&(df['accessories'] == 0)]) == 0 or len(df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal'))&(df['accessories'] == 0)]) == 0
                            ) or (
                        df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar'))&(df['accessories'] == 0), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal'))&(df['accessories'] == 0), 'check'].all()): 
                    
                        person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])
                        continue 
                    else:
                        break
            
                '''
                Filter by palm/back side
                In the training dataset we exclude images with obstructions (accessories) -> to avoid bias
                Finally we take the name of the image
                With .sample we extract # num_train or num_test elements from the dataset and with replace=False we avoid extracting duplicates
                '''
                data_structure[indExp]["labels"].append(0 if df.loc[df["id"] == person_id,'gender'].iloc[0] == "male" else 1)
                '''
                From the entire df dataframe
                we filter on the id of a single person
                I take the palms or backs
                We randomly choose a palm and a hand
                With check == True the image is excluded because it has already been taken
                '''  
                palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar"))&(df['accessories'] == 0)&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
                dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal"))&(df['accessories'] == 0)&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
                
                '''
                The check field indicates that an image has already been taken and therefore cannot be retrieved.
                '''
                df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
                df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True

                data_structure[indExp]["images"].append([palmar_img, dorsal_img])

    print("\nData Preparation Completed\n")
    return data_structure