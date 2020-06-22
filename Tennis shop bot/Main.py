from collections import Counter
from responses import responses, blank_spot
from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity
import spacy

word2vec = spacy.load('en')

exit_commands = ("quit", "goodbye", "exit", "no")

class ChatBot:
# -------------------------------------------------------------------------------------------------------------

    def make_exit(self, user_message):
        for items in exit_commands:
            if items in user_message:
                print("Goodbye!")
                return True
#-------------------------------------------------------------------------------------------------------------

    def chat(self):
        user_message = input("\nWelcome to the tennis shop! How may i help you : ")
        while not self.make_exit(user_message):
            user_message = self.respond(user_message)

# -------------------------------------------------------------------------------------------------------------

    def respond(self, user_message):
        best_response = self.find_intent_match(responses, user_message)
        entity = self.find_entities(user_message)
        print(best_response.format(entity))
        print("\n")
        input_message = input("Can i help you with something else? : ")
        return input_message

# -------------------------------------------------------------------------------------------------------------

    def find_intent_match(self, responses, user_message):
        # tokenizes user input and put counter on it
        bow_user_message = Counter(preprocess(user_message))
        # tokenizes pre-determined responses and adds counter to it
        processed_responses = [Counter(preprocess(response)) for response in responses]
        similarity_list = []
        # check for similar words in user input and responses
        for items in processed_responses:
            similarity_list.append(compare_overlap(items, bow_user_message))
        # chooses index of response with max similar words
        response_index = similarity_list.index(max(similarity_list))
        return responses[response_index]

#-------------------------------------------------------------------------------------------------------------

    def find_entities(self, user_message):
        preprocessed = preprocess(user_message)
        tagged_user_message = pos_tag(preprocessed)
        message_nouns = extract_nouns(tagged_user_message)
        # training vector model with nouns in user message
        tokens = word2vec(" ".join(message_nouns))
        # compare those nouns with the word tennis
        category = word2vec(blank_spot)
        # check for similarity between the nouns in the list and the word tennis
        word2vec_result = compute_similarity(tokens, category)
        # sort the list to get the noun with highest similarity at the end of the list
        word2vec_result.sort(key=lambda x: x[2])
        if len(word2vec_result) < 1:
            return blank_spot
        else:
            return word2vec_result[-1][0]
#-------------------------------------------------------------------------------------------------------------

# initialize ChatBot instance
bot = ChatBot()

bot.chat()

