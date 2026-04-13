import os
import csv
import shutil

base_dir = os.path.abspath(os.path.dirname(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
csv_filepath = os.path.join(dataset_dir, 'asl_mediapipe_keypoints_dataset.csv')
backup_filepath = os.path.join(dataset_dir, 'asl_mediapipe_keypoints_dataset_backup.csv')

def migrate_dataset():
    if not os.path.exists(csv_filepath):
        print(f"Dataset not found at {csv_filepath}. Nothing to migrate.")
        return

    # Check if migration is already done (by checking length of rows)
    with open(csv_filepath, 'r') as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            print("Dataset is empty.")
            return
        
        # 1 label + 126 features = 127 columns
        if len(first_row) >= 127:
            print("Dataset appears to already be in 126-feature (127 columns total) format.")
            return

    # Backup the original
    print(f"Backing up dataset to {backup_filepath}...")
    shutil.copy2(csv_filepath, backup_filepath)

    new_rows = []
    skipped = 0
    with open(csv_filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            
            # Determine original label. Previously it was assumed to be row[0].
            # Original Mediapipe actually put the label at the end (row[-1]).
            try:
                # If the first element is a float, then label is at the end
                float(row[0])
                label = row[-1]
                features = row[:-1]
            except ValueError:
                # If first element is strings, label is at the start
                label = row[0]
                features = row[1:]
                
            if len(features) == 63:
                padded_features = features + ['0.0'] * 63
                new_rows.append([label] + padded_features)
            else:
                skipped += 1

    print(f"Migrating {len(new_rows)} rows. Skipped {skipped} irregular rows.")

    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)
        
    print("Migration complete!")

if __name__ == '__main__':
    migrate_dataset()
