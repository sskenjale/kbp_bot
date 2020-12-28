from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot

chatbot = ChatBot(
    'Shri',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        # 'chatterbot.logic.MathematicalEvaluation',
        # 'chatterbot.logic.TimeLogicAdapter',
        # 'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, I didn\'t get that. I am still learning.',
            'maximum_similarity_threshold': 0.98
        },
        {
            'import_path': 'chatterbot.logic.MathematicalEvaluation',
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)

# Training With Own Questions

trainer = ListTrainer(chatbot)

training_data_quesans = open('training_data/ques_ans.txt').read().splitlines()
training_data_personal = open(
    'training_data/personal_ques.txt').read().splitlines()

training_data = training_data_personal + training_data_quesans

trainer.train(training_data)

# Training With Corpus

trainer_corpus = ChatterBotCorpusTrainer(chatbot)

trainer_corpus.train(
    'chatterbot.corpus.english'
)
