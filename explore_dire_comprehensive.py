import os
import stanza
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

stanza.download('fr')
nlp = stanza.Pipeline('fr')

dsr = "/Users/annacolli/Desktop/1311"
output_dir = "/Users/annacolli/Desktop/dire/clusters/visualizations"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize comprehensive data structures
data_dire = {
    'mood': [],
    'tense': [],
    'person_number': [],
    'iobj_person_number': [],
    'is_interior_discourse': [],
    'subject_has_on': [],
    'discourse_type': [],  # direct, indirect, de_infinitive, none
    'sentence': [],
    'file_name': []
}

# Counters for comprehensive analysis
clusters_by_mood = defaultdict(lambda: defaultdict(int))
clusters_by_tense = defaultdict(lambda: defaultdict(int))
clusters_by_person = defaultdict(lambda: defaultdict(int))
clusters_interior_discourse = defaultdict(int)
clusters_subject_on = defaultdict(int)
clusters_discourse_type = defaultdict(int)
clusters_iobj_person_number = defaultdict(int)  # Track iobj person_number distribution

def extract_grammatical_features(feats):
    """Extract grammatical features from Stanza feats string."""
    features = {}
    if feats:
        for f in feats.split("|"):
            if "=" in f:
                key, value = f.split("=", 1)
                features[key] = value
    return features

def get_person_number(feats):
    """Get person and number from features."""
    features = extract_grammatical_features(feats)
    person = features.get('Person', 'None')
    number = features.get('Number', 'None')
    return f"{person}_{number}"

def get_mood_category(feats):
    """Get the grammatical mood/form category."""
    features = extract_grammatical_features(feats)
    
    # Check VerbForm first for participle and infinitive
    verb_form = features.get('VerbForm')
    if verb_form == 'Inf':
        return 'Inf'
    elif verb_form == 'Part':
        return 'Part'
    
    # Then check Mood for standard moods
    mood = features.get('Mood')
    if mood:
        return mood
    
    return 'Unknown'

def find_subject(verb_word, words):
    """Find the subject of the verb."""
    verb_index = verb_word.id
    for word in words:
        if word.head == verb_index and word.deprel == "nsubj":
            return word
    return None

def find_iobj(verb_word, words):
    """Find indirect object of the verb."""
    verb_index = verb_word.id
    for word in words:
        if word.head == verb_index and word.deprel == "iobj":
            return word
    return None

def check_interior_discourse(verb_word, words, iobj_word):
    """Check if discourse is interior (subject and iobj have same person/number).
    Uses the subject word's person/number features directly."""
    if not iobj_word:
        return False
    
    subject = find_subject(verb_word, words)
    if not subject:
        return False
    
    # Get person/number from the subject word directly
    subject_features = extract_grammatical_features(subject.feats)
    subject_person = subject_features.get('Person')
    subject_number = subject_features.get('Number')
    
    # Get person/number from the indirect object
    iobj_features = extract_grammatical_features(iobj_word.feats)
    iobj_person = iobj_features.get('Person')
    iobj_number = iobj_features.get('Number')
    if  iobj_person == "3" and iobj_number == None and iobj_word.lemma == "soi":
        iobj_person = "3"
        iobj_number = "Sing"
    
    return (subject_person == iobj_person and subject_number == iobj_number)

def check_subject_has_on(verb_word, words):
    """Check if the subject is 'on' for 3rd person contexts."""
    subject = find_subject(verb_word, words)
    if not subject:
        return False
    
    # Check if subject is 'on' and has 3rd person features
    if subject.lemma == "on":
        subject_features = extract_grammatical_features(subject.feats)
        subject_person = subject_features.get('Person')
        return subject_person == '3'
    
    return False

def analyze_discourse_type(verb_word, words, sentence_text):
    """Analyze what follows the verb 'dire' using dependency parsing."""
    verb_index = verb_word.id
    
    # Find all words that depend on this specific "dire" verb
    dependents = [word for word in words if word.head == verb_index]
    
    # Analyze the dependents to determine discourse type
    for dependent in dependents:
        if dependent.text == ":":
            return "direct"
            
        # Check for "que" introducing indirect discourse
        if dependent.lemma == "que" and (dependent.deprel == "mark" or dependent.deprel == "obj"):
            return "indirect"
            
        # Check for "de" + infinitive construction
        if dependent.lemma == "de" and dependent.deprel == "discourse":
            # Look for infinitive that also depends on the main verb
            for word in words:
                if (word.head == verb_index and 
                    "VerbForm=Inf" in (word.feats or "") and
                    word.id > dependent.id):
                    return "de_infinitive"
    
    # If no specific discourse type is found, return default
    return "non_defined"


def save_comprehensive_results():
    """Save all results to files and create visualizations."""
    
    # Save detailed text report
    with open(f'{output_dir}/dire_comprehensive_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE ANALYSIS OF VERB 'DIRE'\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. DISTRIBUTION BY MOOD\n")
        f.write("-" * 30 + "\n")
        for mood, tense_data in clusters_by_mood.items():
            f.write(f"\n{mood.upper()}:\n")
            for tense, count in sorted(tense_data.items()):
                f.write(f"  {tense}: {count}\n")
        
        f.write("\n2. DISTRIBUTION BY TENSE (all moods)\n")
        f.write("-" * 30 + "\n")
        all_tenses = defaultdict(int)
        for mood_data in clusters_by_mood.values():
            for tense, count in mood_data.items():
                all_tenses[tense] += count
        for tense, count in sorted(all_tenses.items()):
            f.write(f"{tense}: {count}\n")
        
        f.write("\n3. DISTRIBUTION BY PERSON_NUMBER\n")
        f.write("-" * 30 + "\n")
        all_persons = defaultdict(int)
        for mood_data in clusters_by_person.values():
            for person, count in mood_data.items():
                all_persons[person] += count
        for person, count in sorted(all_persons.items()):
            f.write(f"{person}: {count}\n")
        
        f.write("\n4. INTERIOR vs NON-INTERIOR DISCOURSE\n")
        f.write("-" * 30 + "\n")
        for disc_type, count in sorted(clusters_interior_discourse.items()):
            f.write(f"{disc_type}: {count}\n")
        
        f.write("\n5. SUBJECT 'ON' ANALYSIS (3rd person singular only)\n")
        f.write("-" * 30 + "\n")
        total_3rd_sing = sum(clusters_subject_on.values())
        for on_type, count in sorted(clusters_subject_on.items()):
            if total_3rd_sing > 0:
                percentage = (count / total_3rd_sing) * 100
                f.write(f"{on_type}: {count} ({percentage:.1f}%)\n")
            else:
                f.write(f"{on_type}: {count}\n")
        
        f.write("\n6. DISCOURSE TYPE FOLLOWING 'DIRE'\n")
        f.write("-" * 30 + "\n")
        # Filter out any None keys before sorting
        valid_discourse_items = {k: v for k, v in clusters_discourse_type.items() if k is not None}
        for disc_type, count in sorted(valid_discourse_items.items()):
            f.write(f"{disc_type}: {count}\n")
        
        f.write("\n8. INDIRECT OBJECT PERSON-NUMBER DISTRIBUTION\n")
        f.write("-" * 45 + "\n")
        # Filter out any None keys before sorting
        valid_iobj_items = {k: v for k, v in clusters_iobj_person_number.items() if k is not None}
        for iobj_person, count in sorted(valid_iobj_items.items()):
            f.write(f"{iobj_person}: {count}\n")
    
    # Create DataFrame for advanced visualizations
    df = pd.DataFrame(data_dire)
    df.to_csv(f'{output_dir}/dire_comprehensive_data.csv', index=False, encoding='utf-8')
    
    print(f"Comprehensive analysis saved to {output_dir}/")

def create_visualizations():
    """Create comprehensive visualizations."""
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Mood distribution
    mood_counts = {}
    for mood, tense_data in clusters_by_mood.items():
        mood_counts[mood] = sum(tense_data.values())
    
    if mood_counts:
        plt.figure(figsize=(12, 6))
        plt.bar(mood_counts.keys(), mood_counts.values())
        plt.title('Distribution of "dire" by Mood/Form')
        plt.xlabel('Mood/Form')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mood_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Discourse type distribution
    if clusters_discourse_type:
        plt.figure(figsize=(10, 6))
        # Filter out any None keys before plotting
        valid_discourse_data = {k: v for k, v in clusters_discourse_type.items() if k is not None}
        plt.bar(valid_discourse_data.keys(), valid_discourse_data.values())
        plt.title('Discourse Type Following "dire"')
        plt.xlabel('Discourse Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/discourse_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Interior vs Non-Interior discourse
    if clusters_interior_discourse:
        plt.figure(figsize=(8, 6))
        plt.bar(clusters_interior_discourse.keys(), clusters_interior_discourse.values())
        plt.title('Interior vs Non-Interior Discourse')
        plt.xlabel('Discourse Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/interior_discourse_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Subject "on" analysis
    if clusters_subject_on:
        plt.figure(figsize=(8, 6))
        plt.bar(clusters_subject_on.keys(), clusters_subject_on.values())
        plt.title('Subject "on" Analysis (3rd person singular only)')
        plt.xlabel('Subject Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/subject_on_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 5. Indirect object person-number distribution
    if clusters_iobj_person_number:
        plt.figure(figsize=(12, 6))
        # Filter out any None keys before plotting
        valid_iobj_data = {k: v for k, v in clusters_iobj_person_number.items() if k is not None}
        plt.bar(valid_iobj_data.keys(), valid_iobj_data.values())
        plt.title('Indirect Object Person-Number Distribution')
        plt.xlabel('iObj Person_Number')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/iobj_person_number_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. Person-Number distribution (all moods combined)
    all_persons = defaultdict(int)
    for mood_data in clusters_by_person.values():
        for person, count in mood_data.items():
            all_persons[person] += count
    
    if all_persons:
        plt.figure(figsize=(10, 6))
        plt.bar(all_persons.keys(), all_persons.values())
        plt.title('Person-Number Distribution (All Moods)')
        plt.xlabel('Person_Number')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/person_number_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 7. Tense distribution (all moods combined)
    all_tenses = defaultdict(int)
    for mood_data in clusters_by_mood.values():
        for tense, count in mood_data.items():
            all_tenses[tense] += count
    
    if all_tenses:
        plt.figure(figsize=(10, 6))
        plt.bar(all_tenses.keys(), all_tenses.values())
        plt.title('Tense Distribution (All Moods)')
        plt.xlabel('Tense')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tense_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 8. Mood-Tense heatmap (for major moods)
    if clusters_by_mood:
        # Create data for heatmap
        mood_tense_matrix = []
        moods = list(clusters_by_mood.keys())
        all_tenses_list = sorted(set(tense for mood_data in clusters_by_mood.values() 
                                    for tense in mood_data.keys()))
        
        for mood in moods:
            row = [clusters_by_mood[mood].get(tense, 0) for tense in all_tenses_list]
            mood_tense_matrix.append(row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(mood_tense_matrix, 
                   xticklabels=all_tenses_list, 
                   yticklabels=moods, 
                   annot=True, 
                   fmt='d', 
                   cmap='YlOrRd')
        plt.title('Mood-Tense Distribution Heatmap')
        plt.xlabel('Tense')
        plt.ylabel('Mood')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mood_tense_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 9. Create DataFrame for correlation analysis
    if data_dire['mood']:
        df = pd.DataFrame(data_dire)
        
        # 9a. Discourse type by mood
        discourse_mood_crosstab = pd.crosstab(df['discourse_type'], df['mood'])
        plt.figure(figsize=(12, 8))
        sns.heatmap(discourse_mood_crosstab, annot=True, fmt='d', cmap='Blues')
        plt.title('Discourse Type by Mood Cross-tabulation')
        plt.xlabel('Mood')
        plt.ylabel('Discourse Type')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/discourse_mood_crosstab.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 9b. Interior discourse by mood
        interior_mood_crosstab = pd.crosstab(df['is_interior_discourse'], df['mood'])
        plt.figure(figsize=(12, 6))
        sns.heatmap(interior_mood_crosstab, annot=True, fmt='d', cmap='Greens')
        plt.title('Interior Discourse by Mood Cross-tabulation')
        plt.xlabel('Mood')
        plt.ylabel('Is Interior Discourse')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/interior_mood_crosstab.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 9c. Person-Number by mood (top combinations)
        person_mood_crosstab = pd.crosstab(df['person_number'], df['mood'])
        plt.figure(figsize=(12, 8))
        sns.heatmap(person_mood_crosstab, annot=True, fmt='d', cmap='Purples')
        plt.title('Person-Number by Mood Cross-tabulation')
        plt.xlabel('Mood')
        plt.ylabel('Person_Number')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/person_mood_crosstab.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 10. Summary statistics pie charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 10a. Mood distribution pie
        axes[0,0].pie(mood_counts.values(), labels=mood_counts.keys(), autopct='%1.1f%%')
        axes[0,0].set_title('Mood Distribution')
        
        # 10b. Discourse type pie
        valid_discourse_data = {k: v for k, v in clusters_discourse_type.items() if k is not None}
        axes[0,1].pie(valid_discourse_data.values(), labels=valid_discourse_data.keys(), autopct='%1.1f%%')
        axes[0,1].set_title('Discourse Type Distribution')
        
        # 10c. Interior discourse pie
        axes[1,0].pie(clusters_interior_discourse.values(), labels=clusters_interior_discourse.keys(), autopct='%1.1f%%')
        axes[1,0].set_title('Interior vs Non-Interior Discourse')
        
        # 10d. Subject 'on' pie  
        axes[1,1].pie(clusters_subject_on.values(), labels=clusters_subject_on.keys(), autopct='%1.1f%%')
        axes[1,1].set_title('Subject "on" Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/summary_pie_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated plots:")
    print("1. mood_distribution.png - Bar chart of mood/form distribution")
    print("2. discourse_type_distribution.png - Bar chart of discourse types")
    print("3. interior_discourse_distribution.png - Interior vs non-interior discourse")
    print("4. subject_on_distribution.png - Subject 'on' analysis")
    print("5. iobj_person_number_distribution.png - Indirect object person-number distribution")
    print("6. person_number_distribution.png - Person-number distribution")
    print("7. tense_distribution.png - Tense distribution")
    print("8. mood_tense_heatmap.png - Mood-tense correlation heatmap")
    print("9. discourse_mood_crosstab.png - Discourse type by mood cross-tabulation")
    print("10. interior_mood_crosstab.png - Interior discourse by mood")
    print("11. person_mood_crosstab.png - Person-number by mood")
    print("12. summary_pie_charts.png - Summary pie charts")
    
    # 11. Advanced cross-dimensional analysis
    if data_dire['mood']:
        df = pd.DataFrame(data_dire)
        
        # 11a. Discourse type by person-number (for direct and indirect discourse)
        df_discourse = df[df['discourse_type'].isin(['direct', 'indirect'])]
        if not df_discourse.empty:
            plt.figure(figsize=(14, 8))
            discourse_person_crosstab = pd.crosstab(df_discourse['discourse_type'], df_discourse['person_number'])
            sns.heatmap(discourse_person_crosstab, annot=True, fmt='d', cmap='Oranges')
            plt.title('Discourse Type by Person-Number (Direct/Indirect only)')
            plt.xlabel('Person_Number')
            plt.ylabel('Discourse Type')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/discourse_person_crosstab.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 11b. Interior discourse by tense
        plt.figure(figsize=(12, 8))
        interior_tense_crosstab = pd.crosstab(df['is_interior_discourse'], df['tense'])
        sns.heatmap(interior_tense_crosstab, annot=True, fmt='d', cmap='Reds')
        plt.title('Interior Discourse by Tense Cross-tabulation')
        plt.xlabel('Tense')
        plt.ylabel('Is Interior Discourse')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/interior_tense_crosstab.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 11c. Subject 'on' by mood (3rd person singular contexts)
        df_3rd_sing = df[df['person_number'] == '3_Sing']
        if not df_3rd_sing.empty:
            plt.figure(figsize=(10, 6))
            on_mood_crosstab = pd.crosstab(df_3rd_sing['subject_has_on'], df_3rd_sing['mood'])
            sns.heatmap(on_mood_crosstab, annot=True, fmt='d', cmap='Greens')
            plt.title('Subject "on" by Mood (3rd person singular contexts only)')
            plt.xlabel('Mood')
            plt.ylabel('Subject has "on"')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/on_mood_crosstab.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 11d. iObj person-number by mood (excluding None)
        df_with_iobj = df[df['iobj_person_number'] != 'None']
        if not df_with_iobj.empty:
            plt.figure(figsize=(12, 8))
            iobj_mood_crosstab = pd.crosstab(df_with_iobj['iobj_person_number'], df_with_iobj['mood'])
            sns.heatmap(iobj_mood_crosstab, annot=True, fmt='d', cmap='Blues')
            plt.title('Indirect Object Person-Number by Mood (iObj present only)')
            plt.xlabel('Mood')
            plt.ylabel('iObj Person_Number')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/iobj_person_mood_crosstab.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 11f. iObj person-number by discourse type (excluding None)
        if not df_with_iobj.empty:
            plt.figure(figsize=(10, 8))
            iobj_discourse_person_crosstab = pd.crosstab(df_with_iobj['iobj_person_number'], df_with_iobj['discourse_type'])
            sns.heatmap(iobj_discourse_person_crosstab, annot=True, fmt='d', cmap='Oranges')
            plt.title('Indirect Object Person-Number by Discourse Type (iObj present only)')
            plt.xlabel('Discourse Type')
            plt.ylabel('iObj Person_Number')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/iobj_person_discourse_crosstab.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 12. Feature correlation matrix (binary features only)
        binary_df = df.copy()
        binary_df['is_interior_num'] = binary_df['is_interior_discourse'].astype(int)
        binary_df['subject_has_on_num'] = binary_df['subject_has_on'].astype(int)
        
        correlation_cols = ['is_interior_num', 'subject_has_on_num']
        if len(correlation_cols) > 1:
            corr_matrix = binary_df[correlation_cols].corr()
            plt.figure(figsize=(6, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       xticklabels=['Interior Discourse', 'Subject "on"'],
                       yticklabels=['Interior Discourse', 'Subject "on"'])
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 13. Detailed statistical summary
        plt.figure(figsize=(16, 12))
        
        # Create a comprehensive summary plot
        gs = plt.GridSpec(3, 3)
        
        # Top person-number combinations
        plt.subplot(gs[0, 0])
        top_persons = df['person_number'].value_counts().head(8)
        plt.bar(range(len(top_persons)), top_persons.values)
        plt.title('Top Person-Number Combinations')
        plt.xticks(range(len(top_persons)), top_persons.index, rotation=45)
        plt.ylabel('Count')
        
        # Top tenses
        plt.subplot(gs[0, 1])
        top_tenses = df['tense'].value_counts().head(8)
        plt.bar(range(len(top_tenses)), top_tenses.values)
        plt.title('Top Tenses')
        plt.xticks(range(len(top_tenses)), top_tenses.index, rotation=45)
        plt.ylabel('Count')
        
        # Mood distribution
        plt.subplot(gs[0, 2])
        mood_dist = df['mood'].value_counts()
        plt.bar(range(len(mood_dist)), mood_dist.values)
        plt.title('Mood Distribution')
        plt.xticks(range(len(mood_dist)), mood_dist.index, rotation=45)
        plt.ylabel('Count')
        
        # Interior discourse by person
        plt.subplot(gs[1, :2])
        interior_person = pd.crosstab(df['person_number'], df['is_interior_discourse'])
        interior_person.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Interior Discourse by Person-Number')
        plt.xlabel('Person_Number')
        plt.ylabel('Count')
        plt.legend(['Non-Interior', 'Interior'])
        plt.xticks(rotation=45)
        
        # Discourse type proportions
        plt.subplot(gs[1, 2])
        discourse_props = df['discourse_type'].value_counts(normalize=True)
        plt.pie(discourse_props.values, labels=discourse_props.index, autopct='%1.1f%%')
        plt.title('Discourse Type Proportions')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("13. discourse_person_crosstab.png - Discourse type by person-number")
    print("14. interior_tense_crosstab.png - Interior discourse by tense")
    print("15. on_mood_crosstab.png - Subject 'on' by mood (3rd person singular)")
    print("16. iobj_person_mood_crosstab.png - iObj person-number by mood")
    print("17. iobj_person_discourse_crosstab.png - iObj person-number by discourse type")
    print("18. feature_correlation_matrix.png - Binary feature correlations")

def create_statistical_report():
    """Create a comprehensive statistical report."""
    if not data_dire['mood']:
        print("No data to create report.")
        return
    
    df = pd.DataFrame(data_dire)
    report_path = f'{output_dir}/statistical_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE ANALYSIS OF 'DIRE' IN FRENCH CORPUS\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write(f"Total occurrences analyzed: {len(df)}\n")
        f.write(f"Number of files processed: {df['file_name'].nunique()}\n")
        f.write(f"Files: {', '.join(df['file_name'].unique())}\n\n")
        
        # Mood distribution
        f.write("MOOD DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        mood_counts = df['mood'].value_counts()
        for mood, count in mood_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{mood}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Tense distribution
        f.write("TENSE DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        tense_counts = df['tense'].value_counts()
        for tense, count in tense_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{tense}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Person-Number distribution
        f.write("PERSON-NUMBER DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        person_counts = df['person_number'].value_counts()
        for person, count in person_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{person}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Discourse analysis
        f.write("DISCOURSE ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        discourse_counts = df['discourse_type'].value_counts()
        for discourse, count in discourse_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{discourse}: {count} ({percentage:.1f}%)\n")
        
        interior_counts = df['is_interior_discourse'].value_counts()
        f.write(f"\nInterior discourse: {interior_counts.get(True, 0)} ({(interior_counts.get(True, 0)/len(df)*100):.1f}%)\n")
        f.write(f"Non-interior discourse: {interior_counts.get(False, 0)} ({(interior_counts.get(False, 0)/len(df)*100):.1f}%)\n\n")
        
        # Indirect object analysis
        f.write("INDIRECT OBJECT ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        # Calculate presence based on iobj_person_number not being 'None'
        has_iobj_series = df['iobj_person_number'] != 'None'
        iobj_counts = has_iobj_series.value_counts()
        f.write(f"With indirect object: {iobj_counts.get(True, 0)} ({(iobj_counts.get(True, 0)/len(df)*100):.1f}%)\n")
        f.write(f"Without indirect object: {iobj_counts.get(False, 0)} ({(iobj_counts.get(False, 0)/len(df)*100):.1f}%)\n\n")
        
        # Subject 'on' analysis
        f.write("SUBJECT 'ON' ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        
        # Focus on 3rd person singular cases only
        third_person_sing_df = df[df['person_number'] == '3_Sing']
        if not third_person_sing_df.empty:
            third_person_on_counts = third_person_sing_df['subject_has_on'].value_counts()
            total_third_person_sing = len(third_person_sing_df)
            on_among_third_sing = third_person_on_counts.get(True, 0)
            not_on_among_third_sing = third_person_on_counts.get(False, 0)
            
            f.write(f"Total 3rd person singular occurrences: {total_third_person_sing}\n")
            f.write(f"With subject 'on': {on_among_third_sing} ({(on_among_third_sing/total_third_person_sing*100):.1f}% of 3rd person singular cases)\n")
            f.write(f"Without subject 'on': {not_on_among_third_sing} ({(not_on_among_third_sing/total_third_person_sing*100):.1f}% of 3rd person singular cases)\n")
        else:
            f.write("No 3rd person singular cases found.\n")
        f.write("\n")
        
        # Detailed iObj analysis
        f.write("DETAILED INDIRECT OBJECT ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        iobj_person_counts = df['iobj_person_number'].value_counts()
        total_with_iobj = len(df[df['iobj_person_number'] != 'None'])
        f.write(f"Total occurrences with indirect object: {total_with_iobj}\n")
        f.write("Person-Number distribution of indirect objects:\n")
        for iobj_person, count in iobj_person_counts.items():
            if iobj_person != 'None':
                percentage_of_total = (count / len(df)) * 100
                percentage_of_iobj = (count / total_with_iobj) * 100 if total_with_iobj > 0 else 0
                f.write(f"  {iobj_person}: {count} ({percentage_of_total:.1f}% of total, {percentage_of_iobj:.1f}% of iObj cases)\n")
        f.write("\n")
        
        # Cross-tabulations
        f.write("KEY CROSS-TABULATIONS:\n")
        f.write("-" * 25 + "\n")
        
        # Mood vs Discourse Type
        f.write("Mood vs Discourse Type:\n")
        mood_discourse = pd.crosstab(df['mood'], df['discourse_type'])
        f.write(mood_discourse.to_string())
        f.write("\n\n")
        
        # Interior discourse vs Mood
        f.write("Interior Discourse vs Mood:\n")
        interior_mood = pd.crosstab(df['is_interior_discourse'], df['mood'])
        f.write(interior_mood.to_string())
        f.write("\n\n")
        
        # Most common patterns
        f.write("MOST COMMON PATTERNS:\n")
        f.write("-" * 25 + "\n")
        pattern_cols = ['mood', 'tense', 'person_number', 'discourse_type']
        patterns = df[pattern_cols].value_counts().head(10)
        f.write("Top 10 Mood-Tense-Person-Discourse combinations:\n")
        for i, (pattern, count) in enumerate(patterns.items(), 1):
            f.write(f"{i}. {pattern}: {count} occurrences\n")
        f.write("\n")
        
        # File-specific statistics
        f.write("FILE-SPECIFIC STATISTICS:\n")
        f.write("-" * 30 + "\n")
        file_stats = df.groupby('file_name').agg({
            'mood': 'count',
            'discourse_type': lambda x: (x == 'direct').sum(),
            'is_interior_discourse': 'sum',
            'iobj_person_number': lambda x: (x != 'None').sum()
        }).rename(columns={
            'mood': 'total_occurrences',
            'discourse_type': 'direct_discourse',
            'is_interior_discourse': 'interior_discourse',
            'iobj_person_number': 'with_iobj'
        })
        f.write(file_stats.to_string())
        f.write("\n\n")
        
        f.write("Report generated successfully.\n")
        f.write(f"Visualization files saved in: {output_dir}/\n")
    
    print(f"Statistical report saved to: {report_path}")

def main():
    """Main processing function."""
    files = [file for file in os.listdir(dsr) if file.endswith(".txt")]
    total_files = len(files)
    
    for idx, file in enumerate(files, 1):
        print(f"Processing file {idx}/{total_files}: {file}")
        
        with open(os.path.join(dsr, file), 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        # Filter sentences containing forms of 'dire'
        dire_pattern = r'\b(?:dire|dis|dit|dits|dite|dites|disons|disais|disait|disaient|disiez|disions|dirai|diras|dira|dirons|direz|diront|dirais|dirait|dirions|diriez|diraient|disant|dites|dise|dises|disions|disiez|disent|dise|dîtes|dît|dîtes|dîmes|dîtes|dîrent)\b'
        dire_sentences = [s for s in sentences if re.search(dire_pattern, s, re.IGNORECASE)]
        
        for dire_sentence in dire_sentences:
            doc = nlp(dire_sentence)
            for sentence in doc.sentences:
                words = sentence.words
                
                for word in words:
                    if word.lemma == "dire":
                        # Extract features
                        features = extract_grammatical_features(word.feats)
                        
                        # Get comprehensive mood category (includes VerbForm)
                        mood = get_mood_category(word.feats)
                        tense = features.get('Tense', 'Unknown')
                        person_number = get_person_number(word.feats)
                        # Handle compound tenses (past participle with auxiliary)
                        if mood == 'Part' and tense == 'Past':
                            # Check for auxiliary verb
                            aux_word = None
                            for w in words:
                                if w.head == word.id and w.upos == "AUX" and w.deprel == "aux:tense":
                                    aux_word = w
                                    break
                            
                            if aux_word:
                                # Get auxiliary's features
                                aux_features = extract_grammatical_features(aux_word.feats)
                                aux_mood = get_mood_category(aux_word.feats)
                                aux_tense = aux_features.get('Tense', 'Unknown')
                                aux_person_number = get_person_number(aux_word.feats)
                                
                                # Apply new compound tense assignment logic
                                if aux_mood == 'Ind' and aux_tense == 'Pres':
                                    global_tense = 'passe_compose'
                                elif aux_mood == 'Ind' and aux_tense == 'Past':
                                    global_tense = 'plus_que_parfait'
                                elif aux_mood in ['Sub', 'Cnd', 'Inf', 'Part'] and aux_tense == 'Pres':
                                    global_tense = 'Past'
                                elif aux_mood in ['Sub', 'Cnd', 'Inf', 'Part'] and aux_tense == 'Past':
                                    global_tense = 'Past'
                                else:
                                    # Fallback for unknown combinations
                                    global_tense = aux_tense
                                
                                # Update variables for compound tense
                                mood = aux_mood
                                tense = global_tense
                                person_number = aux_person_number
                                aux_lemma = aux_word.lemma
                        
                        # Find indirect object
                        iobj_word = find_iobj(word, words)
                        iobj_person_number = get_person_number(iobj_word.feats) if iobj_word else "None"
                        
                        # Check interior discourse (uses auxiliary's person/number for compound tenses)
                        is_interior = check_interior_discourse(word, words, iobj_word)

                        # Check for 'on' subject among 3rd person singular cases only
                        if person_number == "3_Sing":  # Check only 3rd person singular
                            subject_has_on = check_subject_has_on(word, words)
                        else:
                            subject_has_on = False  # Not 3rd person singular, so set to False
                        
                        # Analyze discourse type
                        discourse_type = analyze_discourse_type(word, words, dire_sentence)
                        
                        # Store data
                        data_dire['mood'].append(mood)
                        data_dire['tense'].append(tense)
                        data_dire['person_number'].append(person_number)
                        data_dire['iobj_person_number'].append(iobj_person_number)
                        data_dire['is_interior_discourse'].append(is_interior)
                        data_dire['subject_has_on'].append(subject_has_on)
                        data_dire['discourse_type'].append(discourse_type)
                        data_dire['sentence'].append(dire_sentence)
                        data_dire['file_name'].append(file)
                        
                        # Update counters
                        clusters_by_mood[mood][tense] += 1
                        clusters_by_tense[tense][mood] += 1
                        clusters_by_person[mood][person_number] += 1
                        clusters_interior_discourse["interior" if is_interior else "non_interior"] += 1
                        # Only count "on" subject analysis for 3rd person singular cases
                        if person_number == "3_Sing":
                            clusters_subject_on["has_on" if subject_has_on else "no_on"] += 1
                        clusters_discourse_type[discourse_type] += 1
                        clusters_iobj_person_number[iobj_person_number] += 1  # Track iobj person_number
    
    # Save results and create visualizations
    save_comprehensive_results()
    create_visualizations()
    create_statistical_report()
    
    print(f"Total occurrences of 'dire' analyzed: {len(data_dire['mood'])}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
