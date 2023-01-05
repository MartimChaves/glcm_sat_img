import pandas as pd
import os


def remove_extra_text(row):
    """
    Originally, the image ids are written as:
    "img_000012017.jpg"
    Notice that there is an extra "2017" before ".jpg";
    The goal of this function is to remove that extra year.
    """
    row['image_id'] = row['image_id'][0:-8] + row['image_id'][-4::]
    return row


def main():
    # Load data
    original_df = pd.read_csv("traininglabels.csv")

    # Only select images with a high label confidence score
    # Used .copy to not raise the view/copy warning later when using .apply
    modified_df = original_df[original_df['score'] > 0.8].copy()

    # Originally, the image ids are written as:
    # "img_000012017.jpg"
    # Notice that there is an extra "2017" (sometimes "2018") before ".jpg";
    # The goal of this is to remove that extra year.
    modified_df['image_id'] = modified_df\
        .apply(lambda row: row['image_id'][0:-8] + row['image_id'][-4::],
               axis=1)

    # Check which images actually exist
    available_imgs_lst = os.listdir("train_images")

    # Only keep those that exist
    modified_df = modified_df[modified_df['image_id']
                              .isin(available_imgs_lst)]

    # Drop duplicates
    modified_df = modified_df.drop_duplicates(subset='image_id')
    # It's odd that there are duplicates, but EDA seems to show that there
    # indeed differences between classes, so I'm assuming that the images in
    # the train_images folder are with respect to the earliest year, 2017.
    # No clarifying information could be found in the WiDS data description,
    # hence the need for this assumption.
    # Plus, manually looking at the image ids that remain, it seems that they
    # match the names of the available images quite well (starts with image 1
    # and ends with image 10999).

    # Drop the 'score' column
    modified_df = modified_df.drop(columns='score')

    # Rename columns to easier to use terms
    modified_df = modified_df.rename(columns={'image_id': 'image',
                                              'has_oilpalm': 'label'})

    # Save df to new csv file
    modified_df.to_csv("train.csv", index=False)


if __name__ == "__main__":
    main()
