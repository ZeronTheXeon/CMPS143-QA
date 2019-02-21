from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk
from nltk.stem import WordNetLemmatizer

base = __import__('baseline-stub')

GRAMMAR = """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

chunker = nltk.RegexpParser(GRAMMAR)
lmtzr = nltk.stem.WordNetLemmatizer()
LOC_PP = {"in", "on", "at", "to"}
WHY_PP = {"because", "as", "for", "to", "so"}
WHAT_PP = {"the", "a", "that", "is"}


# AVERAGE RECAL =     0.4671
# AVERAGE PRECISION = 0.4309
# AVERAGE F-MEASURE = 0.4191

stopwords = set(nltk.corpus.stopwords.words("english"))
stopwords.union({"who", "what", "when", "where", "why"})
stopwords = stopwords - LOC_PP - WHY_PP - WHAT_PP


def loc_filter(subtree):
    return subtree.label() == "PP"


def why_filter(subtree):
    return subtree.label() == "VP"


def what_filter(subtree):
    return subtree.label() == "NP" or subtree.label() == "VP"


def get_answer_phrase(question, sentence):
    """
    Given a question and the sentence with an answer, extract the answer.
    :param question: Question asked of us.
    :param sentence: Sentence with question in it.
    :return: string phrase of answer
    """

    # Tokenize question for W-word
    q_toks = nltk.word_tokenize(question)
    sent_toks = nltk.word_tokenize(sentence)
    sent_pos = nltk.pos_tag(sent_toks)
    sent_pos = [(x[0].lower(), x[1]) for x in sent_pos]
    tree = chunker.parse(sent_pos)

    q_toks = [word.lower() for word in q_toks]

    if "where" in q_toks:
        # print("q_toks contains where!", q_toks)
        set_to_use = LOC_PP
        filter_to_use = loc_filter
    elif "why" in q_toks:
        # print("q_toks contains why!", q_toks)
        set_to_use = WHY_PP
        filter_to_use = why_filter
    else:
        # print("else!", q_toks)
        set_to_use = WHAT_PP
        filter_to_use = what_filter

    answer_list = []
    # print(tree)
    for subtree in tree.subtrees(filter=filter_to_use):
        # print(subtree)
        if subtree[0][0] in set_to_use:
            # print("appending", subtree[0][0][0], " as it is in set_to_use")
            answer_list.append(subtree)
            # print("1:      ", answer_list)

    final_answer = " ".join([token[0] for token in answer_list[0].leaves()]) \
        if len(answer_list) > 0 else sentence
    return final_answer.strip()


def get_answer_sentence(question, story):
    version = question["type"]

    question_text = question["text"]

    # prefer scheherazade
    if "Sch" in version:
        text = story['sch']
    else:
        text = story['text']

    question_sent = base.get_sentences(question_text)

    text = base.get_sentences(text)

    answer = base.baseline(question_sent[0], text, stopwords)

    answer_text = ""
    for (x, y) in answer:
        answer_text += (" " if x[0].isalnum() else "") + x
    # print("Difficulty: ", question['difficulty'] + "\n")
    # print("Question:", question_text + "\n")
    # print("Answer:", answer_text + "\n")

    return question_text, answer_text


def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    ###     Your Code Goes Here         ###

    ###     End of Your Code         ###
    question_text, answer_text = get_answer_sentence(question, story)

    answer = get_answer_phrase(question_text, answer_text)
    # print("Extracted Answer:", answer + "\n\n")
    return answer


#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa(evaluate=False):
    QA = QAEngine(evaluate=evaluate)
    QA.run()
    QA.save_answers()


#############################################################


def main():
    run_qa(evaluate=True)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    # score_answers()


if __name__ == "__main__":
    wordnet_lemmatizer = WordNetLemmatizer()
    # print(wordnet_lemmatizer.lemmatize("had", pos='v'))
    # print(wordnet_lemmatizer.lemmatize("have", pos='v'))
    main()
