#ifndef SPAM_CLASSIFIER_H
#define SPAM_CLASSIFIER_H

#define TABLE_SIZE 1000
#define MAX_WORD_LENGTH 50
#define MAX_EMAIL_LENGTH 1000

typedef struct HashNode {
    char word[MAX_WORD_LENGTH];
    int spam_count;
    int ham_count;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode* table[TABLE_SIZE];
    int total_spam_emails;
    int total_ham_emails;
    int total_spam_words;
    int total_ham_words;
} Vocabulary;

// Function declarations
unsigned int hash(const char* word);
HashNode* create_hash_node(const char* word);
void init_vocabulary(Vocabulary* vocab);
void clean_word(char* word);
void train_classifier(Vocabulary* vocab, const char* email, int is_spam);
double calculate_probability(Vocabulary* vocab, const char* email);
void free_vocabulary(Vocabulary* vocab);
void load_training_data(Vocabulary* vocab, const char* filename);
void save_model(Vocabulary* vocab, const char* filename);
void load_model(Vocabulary* vocab, const char* filename);
int get_vocabulary_size(Vocabulary* vocab);

#endif