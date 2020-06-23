"""
Pos tags descriptions
"""
def pos_dict_en():
    """
    Source:
    Taylor, Ann & Marcus, Mitchell & Santorini, Beatrice. (2003).
    The Penn Treebank: An overview. 10.1007/978-94-010-0201-1_1. 
    """

    return {'CC': 'Coordinating conj.',
            'TO': 'Infinitival "to"',
            'CD': 'Cardinal number',
            'UH': 'Interjection',
            'DT': 'Determiner',
            'VB': 'Verb (base form)',
            'EX': 'Existential there',
            'VBD': 'Verb (past tense)',
            'FW': 'Foreign word',
            'VBG': 'Verb (gerund/present pple)',
            'IN': 'Preposition',
            'VBN': 'Verb (past participle)',
            'JJ': 'Adjective',
            'VBP': 'Verb (non-3rd ps. sg. present)',
            'JJR': 'Adjective (comparative)',
            'VBZ': 'Verb (3rd ps. sg. present)',
            'JJS': 'Adjective (superlative)',
            'WDT': 'Wh-determiner',
            'LS': 'List item marker',
            'WP': 'Wh-pronoun',
            'MD': 'Modal',
            'WP$': 'Possessive wh-pronoun',
            'NN': 'Noun (singular or mass)',
            'WRB': 'Wh-adverb',
            'NNS': 'Noun (plural)',
            '#': 'Pound sign',
            'NNP': 'Proper noun, singular',
            '$': 'Dollar sign',
            'PRP$': 'Possessive pronoun',
            'NNPS': 'Proper noun (plural)',
            '.': 'Sentence-final punctuation',
            'PDT': 'Predeterminer', 
            ',': 'Comma',
            'P': 'OS Possessive ending',
            ':': 'Colon / semi-colon',
            'PRP': 'Personal pronoun',
            '(': 'Left bracket character',
            'PP$': 'Possessive pronoun',
            ')': 'Right bracket character',
            'RB': 'Adverb',
            '"': 'Straight double quote',
            'RBR': 'Adverb (comparative)',
            '‘': 'Left open single quote',
            'RBS': 'Adverb (superlative)',
            '“': 'Left open double quote',
            'RP': 'Particle',
            '’': 'Right close single quote',
            'SYM': 'Symbol',
            '”': 'Right close double quo',
            '<s>': 'Start of sentence',
            'POS': 'POS',
            '``': '``',
            '-NONE-': 'None',
            '</s>': 'End of sentence',
            '<UNK>': 'Unknown',
            "''": 'Empty tag'
    }

def pos_dict_sv(): #UPOS
    return {'ADJ': 'Adjective',
            'ADP': 'Adposition',
            'ADV': 'Adverb',
            'AUX': 'Auxiliary',
            'CCONJ': 'Coordinating conjunction',
            'DET': 'Determiner',
            'INTJ': 'Interjection',
            'NOUN': 'Noun',
            'NUM': 'Numeral',
            'PART': 'Particle',
            'PRON': 'Pronoun',
            'PROPN': 'Proper noun',
            'PUNCT': 'Punctuation',
            'SCONJ': 'Subordinating conjunction',
            'SYM': 'Symbol',
            'VERB': 'Verb',
            'X': 'Other',
            '<UNK>': 'Unknown',
            '<s>': 'Start of sentence',
            '</s>': 'End of sentence'
    }

def pos_dict_sv_suc():
    """
    Source: https://spraakbanken.gu.se/parole/Docs/SUC2.0-manual.pdf
    """
    dict_1 = pos_dict_en()
    dict_1.update({'NN': 'Noun',
            'VB': 'Verb',
            'PM': 'Proper Noun',
            'PP': 'Preposition',
            'MAD': 'Major delimiter',
            'HP': 'Interrogative/Relative Pronoun',
            'AB': 'Adverb',
            'RO': 'Ordinal number',
            'MID': 'Minor delimiter',
            'RG': 'Cardinal number',
            'PS': 'Possessive',
            'PC': 'Participle',
            'PAD': 'Pairwise delimiter',
            'KN': 'Conjunction',
            'PN': 'Pronoun',
            'HA': 'Interrogative/Relative Adverb',
            'IE': 'Infinitive Marker',
            'PL': 'Particle',
            'SN': 'Subjunction',
            'UO': 'Foreign Word',
            'HD': 'Interrogative/Relative Determiner',
            'HS': 'Interrogative/Relative Possessive',
    })
    return dict_1