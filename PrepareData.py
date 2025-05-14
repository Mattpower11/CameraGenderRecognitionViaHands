import pandas as pd
import numpy as np

def prepare_data(csv_path:str, num_exp: int, num_train: int, num_test: int):
    # Load the data from csv metadata file
    df = pd.read_csv(csv_path)
    # Create a data structure to store the images' name and the corresponding label
    data_structure = {}
    
    print("Preparing Data\n")

    # Populate the data structure
    for indExp in range(num_exp):
        print(f"\tExp {indExp}")
        data_structure[indExp] = {}
        df['check'] = False
        data_structure[indExp]['train'], df = prepare_data_train(num_train= num_train, df = df)
        data_structure[indExp]['test']= prepare_data_test(num_test= num_test, df=df)  
    
    print("Data Preparation Completed\n")
    return data_structure

def prepare_data_train(num_train: int,  df: pd.DataFrame ):
    result_dict = {
                "labels": [],
                "images": []
            }
    
    gender = ['male',  'female']

    print("\t\tTraining")

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
            result_dict["labels"].append(0 if df.loc[df["id"] == person_id,'gender'].iloc[0] == "male" else 1)
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

            result_dict["images"].append([palmar_img, dorsal_img])
    return result_dict, df

def prepare_data_test(num_test: int, df: pd.DataFrame):
    result_dict = {
        "labels": [],
        "images": []
    } 
    
    male_female_list = ['male', 'female']

    print("\t\tTesting\n")

    for gender in male_female_list:
        person_id_list = df.loc[(df['gender'] == gender), 'id'].unique()

        for person_id in person_id_list:
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

        for _ in range(num_test):
            person_id = np.random.choice(person_id_list)
            '''
            Filter by palm/back side
            In the training dataset we exclude images with obstructions (accessories) -> to avoid bias
            Finally we take the name of the image
            With .sample we extract # num_train or num_test elements from the dataset and with replace=False it avoids extracting duplicates
            '''
            result_dict["labels"].append(0 if df.loc[df["id"] == person_id,'gender'].iloc[0] == "male" else 1)
            '''
            From the entire dataframe df
            we filter on a single person id
            I take the palms or backs
            We randomly choose a palm and a hand
            '''

            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
           
            '''
            The check field indicates that an image has already been taken and therefore cannot be retrieved.            
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True
            

            result_dict["images"].append([palmar_img, dorsal_img])

            '''
                Exclude people who no longer have palm and back images to extract
            '''
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

    return result_dict