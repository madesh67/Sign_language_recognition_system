# src/delete_label_data.py
import csv
import os
import argparse
from collections import Counter

def delete_label_samples(csv_path, label_to_delete, backup=True):
    """Remove all samples of a specific label from CSV"""
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    # Backup original
    if backup:
        backup_path = csv_path + ".backup"
        import shutil
        shutil.copy2(csv_path, backup_path)
        print(f"✓ Backup created: {backup_path}")
    
    # Read all rows
    rows = []
    deleted_count = 0
    label_counts = Counter()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = row[0]
            label_counts[label] += 1
            
            if label != label_to_delete:
                rows.append(row)
            else:
                deleted_count += 1
    
    # Write cleaned data
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"\n✓ Deleted {deleted_count} samples of label '{label_to_delete}'")
    print(f"✓ Remaining samples: {len(rows)}")
    print(f"\nLabel distribution after deletion:")
    remaining_counts = Counter(row[0] for row in rows)
    for label, count in sorted(remaining_counts.items()):
        print(f"  {label}: {count}")

def list_labels(csv_path):
    """Show all labels and their counts"""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    label_counts = Counter()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                label_counts[row[0]] += 1
    
    print(f"\nLabels in {csv_path}:")
    print(f"{'Label':<20} Count")
    print("-" * 30)
    for label, count in sorted(label_counts.items()):
        print(f"{label:<20} {count}")
    print(f"\nTotal samples: {sum(label_counts.values())}")
    print(f"Total labels: {len(label_counts)}")

def main():
    ap = argparse.ArgumentParser(description="Delete wrong label data from CSV")
    ap.add_argument("--csv", type=str, default="data/samples.csv", help="Path to samples CSV")
    ap.add_argument("--delete", type=str, default="", help="Label to delete")
    ap.add_argument("--list", action="store_true", help="List all labels and counts")
    ap.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    args = ap.parse_args()
    
    if args.list:
        list_labels(args.csv)
    elif args.delete:
        delete_label_samples(args.csv, args.delete, backup=not args.no_backup)
        print(f"\n⚠ Run 'python src/train.py' to retrain the model")
    else:
        print("Use --list to see labels or --delete LABEL to remove a label")

if __name__ == "__main__":
    main()
