from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk
from nltk.stem import WordNetLemmatizer

import baseline_stub as base
import constituency_demo_stub as constit

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

# These stopwords are used for the better sentene recall. Overall it drops our F stop so when 
# we are using chunking, set the baseline call to use stopwords not stopwrods1
stopwords1 = set(nltk.corpus.stopwords.words("english"))
stopwords1.union({"who", "what", "when", "where", "why", "'s"})
stopwords1 = stopwords1 - {"had", "have", "from"}


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
        text_backup = story['sch']
    else:
        text = story['text']
        text_backup = story['text']

    text_backup = nltk.sent_tokenize(text_backup)
    question_sent = base.get_sentences(question_text)

    text = base.get_sentences(text)

    # Use stopwords instead of stopwords1 when using chunking and not depend trees
    answer, sent_number = base.baseline(question_sent[0], text, stopwords1)

    answer_text = ""
    for (x, y) in answer:
        answer_text += (" " if x[0].isalnum() else "") + x

    # print("Difficulty: ", question['difficulty'] + "\n")
    # print("Question:", question_text + "\n")
    # print("Answer:", answer_text + "\n")

    return question_text, text_backup[sent_number], sent_number


def get_dependencies(question, story, sent_num):
    version = question["type"]
    print(version)

    if "Sch" in version:
        story_dep = story['sch_dep'][sent_num]


    else:
        story_dep = story['story_dep'][sent_num]

    question_dep = question["dep"]

    return question_dep, story_dep


def get_constituency(question, story, sent_num):
    version = question["type"]

    if "Sch" in version:
        story_con = story['sch_par'][sent_num]


    else:
        story_con = story['story_par'][sent_num]

    question_con = question["par"]

    return question_con, story_con


def find_answer_con(qcon, scon, question, q, s):
    (word, rest) = question.split(maxsplit=1)
    word = word.lower()
    # print("Word is", word, "for question", question)
    # AVERAGE RECALL =    0.4938
    # AVERAGE PRECISION = 0.4513
    # AVERAGE F-MEASURE = 0.4164
    # answer = get_answer(q, s, True)
    # return answer
    # print("Sentence Con is", scon)
    if word == "who":
        pattern = nltk.ParentedTree.fromstring("(NP)")
        sub_pattern = nltk.ParentedTree.fromstring("(NP)")
    elif word == "what":
        pattern = nltk.ParentedTree.fromstring("(VP)")
        sub_pattern = nltk.ParentedTree.fromstring("(NP)")
    elif word == "did" or word == "does":
        if "no" in question:
            return "no"
        return "yes"
    elif word == "had":
        if "no" in question:
            return "yes"
        return "no"
    elif word == "when":
        # mc500.train.23.1 has WHAVP, please investigate
        if constit.pattern_matcher("(WHAVP)", qcon) is not None:
            pattern = nltk.ParentedTree.fromstring("(VP (*) (NP))")
            sub_pattern = nltk.ParentedTree.fromstring("(NP)")
        else:
            pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")
            sub_pattern = nltk.ParentedTree.fromstring("(PP)")
    elif word == "where":
        # AVERAGE RECALL =    0.6700
        # AVERAGE PRECISION = 0.5086
        # AVERAGE F-MEASURE = 0.5317
        # if "to" in question[:15]:
        #     pattern = nltk.ParentedTree.fromstring("(TO (*))")
        #     sub_pattern = nltk.ParentedTree.fromstring("(LOC)")
        # else:
        if constit.pattern_matcher("(SINV)", scon) is not None:
            pattern = nltk.ParentedTree.fromstring("(PP)")
            sub_pattern = nltk.ParentedTree.fromstring("(PP)")
        else:
            pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")
            sub_pattern = nltk.ParentedTree.fromstring("(NP)")
    elif word == "why":
        pattern = nltk.ParentedTree.fromstring("(SBAR)")
        sub_pattern = nltk.ParentedTree.fromstring("(SBAR)")
    elif word == "how":
        pattern = nltk.ParentedTree.fromstring("(VP)")
        sub_pattern = nltk.ParentedTree.fromstring("(RB)")
    else:
        pattern = nltk.ParentedTree.fromstring("(VP (*))")
        sub_pattern = nltk.ParentedTree.fromstring("(*)")

    sub_tree = constit.pattern_matcher(pattern, scon)

    if sub_tree is None:
        answer = get_answer(q, s, True)
        return answer

    sub_tree_2 = constit.pattern_matcher(sub_pattern, sub_tree)
    if sub_tree_2 is None:
        # print("sub_tree_2 is None for question", question, "con of", scon)
        # sub_tree_2 = sub_tree
        answer = get_answer(q, s, True)
        return answer

    answer = " ".join(sub_tree_2.leaves())
    return answer


def get_answer(question, story, fail=False):
    """
    :param question: dict
    :param story: dict
    :param fail: bool True if answer couldnt be found in cons
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
    question_text, answer_text, sent_number = get_answer_sentence(question, story)

    # print("For question", question_text, "got sentence", answer_text)
    # part 2 with dependcy
    # qdep, sdep = get_dependencies(question, story, sent_number)
    # answer = depend.find_answer(qgraph, sgraph)

    if fail:
        answer = get_answer_phrase(question_text, answer_text)
    else:
        qcon, scon = get_constituency(question, story, sent_number)
        answer = find_answer_con(qcon, scon, question_text, question, story)

    (word, rest) = question_text.split(maxsplit=1)
    word = word.lower()
    if word == "how":
        question_words = question_text.lower()
        question_words = question_words.split()
        answer_words = answer.split()
        print(question_words, answer_words)
        resultWords = [word for word in answer_words if word.lower() not in question_words]
        answer = ' '.join(resultWords)

    # print("Extracted Answer:", answer + "\n\n")
    return answer.lower().strip()


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
    run_qa(evaluate=False)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()


if __name__ == "__main__":
    wordnet_lemmatizer = WordNetLemmatizer()
    # print(wordnet_lemmatizer.lemmatize("had", pos='v'))
    # print(wordnet_lemmatizer.lemmatize("have", pos='v'))
    main()
