import os
import jiwer
from g2p_en import G2p

def load_transcripts(base_dir):
    """Recursively load all LibriSpeech transcripts into a dict: {file_id: text}"""
    transcripts = {}
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.trans.txt'):
                with open(os.path.join(root, f), 'r') as fp:
                    for line in fp:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            transcripts[file_id] = text
    return transcripts

def main():
    tsv_path = 'data/clustering/librispeech/valid.tsv'
    libri_dir = 'data/LibriSpeech/dev-clean'
    units_file = 'data/transcription_phones/valid_units.txt'
    
    print('Loading transcripts...')
    transcripts_map = load_transcripts(libri_dir)
    
    g2p = G2p()
    compact = True # wav2vec-U compacts phones
    
    reference_words = []
    reference_phones = []
    hypothesis_phones = []
    
    print('Reading true labels...')
    unique_words = set()
    with open(tsv_path, 'r') as f:
        lines = f.read().splitlines()
        root_dir = lines[0] # first line is root dir
        for line in lines[1:]:
            rel_path, _ = line.split('\t')
            basename = os.path.basename(rel_path)
            file_id = os.path.splitext(basename)[0]
            
            true_text = transcripts_map.get(file_id, "")
            reference_words.append(true_text)
            
            # Convert true text to phones
            words = true_text.split()
            phones = []
            for w in words:
                unique_words.add(w)
                ph = g2p(w)
                if compact:
                    ph = [p[:-1] if p[-1].isnumeric() else p for p in ph]
                phones.extend(ph)
            reference_phones.append(" ".join(phones))

    # Generate Lexicon for flashlight decoding
    lexicon_path = 'data/text/phones/lexicon_filtered.lst'
    print(f'Writing lexicon to {lexicon_path} ...')
    with open(lexicon_path, 'w') as lf:
        for w in sorted(list(unique_words)):
            ph = g2p(w)
            if compact:
                ph = [p[:-1] if p[-1].isnumeric() else p for p in ph]
            lf.write(f"{w} {' '.join(ph)}\n")

    print('Reading hypotheses (Viterbi outputs)...')
    with open(units_file, 'r') as f:
        # valid_units.txt has one hypothesis phone sequence per line
        hypothesis_phones = [line.strip() for line in f.read().splitlines()]
        
    print('Reading word hypotheses (KenLM outputs)...')
    words_file = 'data/transcription_words/valid.txt'
    hypothesis_words = []
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            hypothesis_words = [line.strip() for line in f.read().splitlines()]
    else:
        print(f"Warning: {words_file} not found. Run generator with kenlm to get words.")
    
    # Check length match
    if len(reference_phones) != len(hypothesis_phones):
        print(f"Warning: {len(reference_phones)} references but {len(hypothesis_phones)} hypotheses!")
        min_len = min(len(reference_phones), len(hypothesis_phones))
        reference_phones = reference_phones[:min_len]
        hypothesis_phones = hypothesis_phones[:min_len]
        reference_words = reference_words[:min_len]
        if hypothesis_words:
            hypothesis_words = hypothesis_words[:min_len]
        
    print(f'Calculating metrics for {len(reference_phones)} files...')
    
    print(f"==================================================")
    # 1. Phone Error Rate (PER)
    try:
        per = jiwer.wer(reference_phones, hypothesis_phones)
        print(f"Phone Error Rate (PER): {per * 100:.2f}%")
    except Exception as e:
        print(f"Failed to calculate PER: {e}")
        
    # 2. Word Error Rate (WER)
    if hypothesis_words:
        try:
            wer = jiwer.wer(reference_words, hypothesis_words)
            print(f"Word Error Rate  (WER): {wer * 100:.2f}%")
        except Exception as e:
            print(f"Failed to calculate WER: {e}")
            
    print(f"==================================================")

if __name__ == '__main__':
    main()
