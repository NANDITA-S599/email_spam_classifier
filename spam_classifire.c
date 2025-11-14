#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "spam_classifier.h"

// Hash function for string
unsigned int hash(const char* word) {
    unsigned long hash = 5381;
    int c;
    while ((c = *word++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

// Create new hash node
HashNode* create_hash_node(const char* word) {
    HashNode* new_node = (HashNode*)malloc(sizeof(HashNode));
    if (new_node == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    strcpy(new_node->word, word);
    new_node->spam_count = 0;
    new_node->ham_count = 0;
    new_node->next = NULL;
    return new_node;
}

// Initialize vocabulary
void init_vocabulary(Vocabulary* vocab) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        vocab->table[i] = NULL;
    }
    vocab->total_spam_emails = 0;
    vocab->total_ham_emails = 0;
    vocab->total_spam_words = 0;
    vocab->total_ham_words = 0;
}

// Clean and tokenize text
void clean_word(char* word) {
    int i = 0, j = 0;
    while (word[i]) {
        if (isalnum(word[i])) {
            word[j++] = tolower(word[i]);
        }
        i++;
    }
    word[j] = '\0';
}

// Train the classifier with an email
void train_classifier(Vocabulary* vocab, const char* email, int is_spam) {
    char copy[MAX_EMAIL_LENGTH];
    strcpy(copy, email);
    
    char* word = strtok(copy, " \t\n\r");
    int word_count = 0;
    
    while (word != NULL) {
        char clean[MAX_WORD_LENGTH];
        strcpy(clean, word);
        clean_word(clean);
        
        // Skip empty words and very short words
        if (strlen(clean) > 2) {
            unsigned int index = hash(clean);
            HashNode* current = vocab->table[index];
            HashNode* prev = NULL;
            
            // Search for existing word
            while (current != NULL && strcmp(current->word, clean) != 0) {
                prev = current;
                current = current->next;
            }
            
            if (current == NULL) {
                // Word not found, create new node
                current = create_hash_node(clean);
                if (prev == NULL) {
                    vocab->table[index] = current;
                } else {
                    prev->next = current;
                }
            }
            
            // Update counts - NAIVE BAYES IMPLEMENTATION
            if (is_spam) {
                current->spam_count++;
                vocab->total_spam_words++;
            } else {
                current->ham_count++;
                vocab->total_ham_words++;
            }
            word_count++;
        }
        
        word = strtok(NULL, " \t\n\r");
    }
    
    // Update total email counts
    if (is_spam) {
        vocab->total_spam_emails++;
    } else {
        vocab->total_ham_emails++;
    }
}

// Calculate spam probability for an email - CORE NAIVE BAYES ALGORITHM
double calculate_probability(Vocabulary* vocab, const char* email) {
    if (vocab->total_spam_emails == 0 || vocab->total_ham_emails == 0) {
        return 0.5; // Neutral if no training data
    }
    
    // Prior probabilities P(spam) and P(ham)
    double prior_spam = (double)vocab->total_spam_emails / 
                       (vocab->total_spam_emails + vocab->total_ham_emails);
    double prior_ham = 1.0 - prior_spam;
    
    // Use log probabilities to avoid floating point underflow
    double log_prob_spam = log(prior_spam);
    double log_prob_ham = log(prior_ham);
    
    char copy[MAX_EMAIL_LENGTH];
    strcpy(copy, email);
    char* word = strtok(copy, " \t\n\r");
    int words_processed = 0;
    
    int vocab_size = get_vocabulary_size(vocab);
    
    while (word != NULL) {
        char clean[MAX_WORD_LENGTH];
        strcpy(clean, word);
        clean_word(clean);
        
        if (strlen(clean) > 2) {
            unsigned int index = hash(clean);
            HashNode* current = vocab->table[index];
            
            // Find the word in vocabulary
            while (current != NULL && strcmp(current->word, clean) != 0) {
                current = current->next;
            }
            
            // Calculate P(word|spam) and P(word|ham) with Laplace smoothing
            double p_word_spam, p_word_ham;
            
            if (current != NULL) {
                // Word found in vocabulary
                p_word_spam = (current->spam_count + 1.0) / 
                             (vocab->total_spam_words + vocab_size);
                p_word_ham = (current->ham_count + 1.0) / 
                            (vocab->total_ham_words + vocab_size);
            } else {
                // Unknown word - use Laplace smoothing
                p_word_spam = 1.0 / (vocab->total_spam_words + vocab_size);
                p_word_ham = 1.0 / (vocab->total_ham_words + vocab_size);
            }
            
            // Add to log probabilities (multiplication in normal space = addition in log space)
            log_prob_spam += log(p_word_spam);
            log_prob_ham += log(p_word_ham);
            words_processed++;
        }
        
        word = strtok(NULL, " \t\n\r");
    }
    
    if (words_processed == 0) return 0.5;
    
    // Convert back from log space and apply Bayes theorem
    // P(spam|email) = P(email|spam) * P(spam) / [P(email|spam)*P(spam) + P(email|ham)*P(ham)]
    double prob_spam_given_email = exp(log_prob_spam) / 
                                  (exp(log_prob_spam) + exp(log_prob_ham));
    
    return prob_spam_given_email;
}

// Get total vocabulary size
int get_vocabulary_size(Vocabulary* vocab) {
    int size = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        HashNode* current = vocab->table[i];
        while (current != NULL) {
            size++;
            current = current->next;
        }
    }
    return size;
}

// Free memory
void free_vocabulary(Vocabulary* vocab) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        HashNode* current = vocab->table[i];
        while (current != NULL) {
            HashNode* temp = current;
            current = current->next;
            free(temp);
        }
        vocab->table[i] = NULL;
    }
}

// Load training data from file
void load_training_data(Vocabulary* vocab, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open training file %s\n", filename);
        return;
    }
    
    char line[MAX_EMAIL_LENGTH];
    int line_count = 0;
    
    printf("Loading training data...\n");
    
    while (fgets(line, sizeof(line), file)) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;
        
        // Skip empty lines
        if (strlen(line) < 3) continue;
        
        // First character indicates class: 1=spam, 0=ham
        int is_spam = (line[0] == '1');
        const char* email_content = line + 2; // Skip class indicator and space
        
        if (strlen(email_content) > 0) {
            train_classifier(vocab, email_content, is_spam);
            line_count++;
        }
    }
    
    fclose(file);
    printf("Loaded %d training emails\n", line_count);
    printf("Spam emails: %d, Ham emails: %d\n", 
           vocab->total_spam_emails, vocab->total_ham_emails);
    printf("Vocabulary size: %d words\n", get_vocabulary_size(vocab));
}

// Save trained model to file
void save_model(Vocabulary* vocab, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not create model file %s\n", filename);
        return;
    }
    
    // Save header information
    fprintf(file, "%d %d %d %d\n", 
            vocab->total_spam_emails, 
            vocab->total_ham_emails,
            vocab->total_spam_words,
            vocab->total_ham_words);
    
    // Save vocabulary
    for (int i = 0; i < TABLE_SIZE; i++) {
        HashNode* current = vocab->table[i];
        while (current != NULL) {
            fprintf(file, "%s %d %d\n", 
                    current->word, 
                    current->spam_count, 
                    current->ham_count);
            current = current->next;
        }
    }
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load trained model from file
void load_model(Vocabulary* vocab, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open model file %s\n", filename);
        return;
    }
    
    init_vocabulary(vocab);
    
    // Read header
    fscanf(file, "%d %d %d %d", 
           &vocab->total_spam_emails, 
           &vocab->total_ham_emails,
           &vocab->total_spam_words,
           &vocab->total_ham_words);
    
    char word[MAX_WORD_LENGTH];
    int spam_count, ham_count;
    
    while (fscanf(file, "%s %d %d", word, &spam_count, &ham_count) == 3) {
        unsigned int index = hash(word);
        HashNode* current = vocab->table[index];
        HashNode* prev = NULL;
        
        while (current != NULL && strcmp(current->word, word) != 0) {
            prev = current;
            current = current->next;
        }
        
        if (current == NULL) {
            current = create_hash_node(word);
            if (prev == NULL) {
                vocab->table[index] = current;
            } else {
                prev->next = current;
            }
        }
        
        current->spam_count = spam_count;
        current->ham_count = ham_count;
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    printf("Vocabulary size: %d words\n", get_vocabulary_size(vocab));
}