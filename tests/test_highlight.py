from src.highlight import best_sentence, claims_by_source

PASSAGE = (
    "Introduction to the cohort. The dataset includes mRNAseq data from 105 controls and 61 subjects "
    "with late-stage AMD. Samples were sequenced on an Illumina platform."
)


def test_claims_by_source_maps_sources_to_their_sentences():
    raw = "The study had 105 controls [S1]. Sequencing used Illumina [S2, S3]. No citation here."
    claims = claims_by_source(raw)
    assert claims[1] == "The study had 105 controls [S1]."
    assert claims[2] == claims[3] == "Sequencing used Illumina [S2, S3]."
    assert 4 not in claims


def test_best_sentence_finds_the_supporting_sentence():
    sentence = best_sentence(PASSAGE, "The study included 105 controls and 61 AMD subjects [S1].")
    assert sentence.startswith("The dataset includes mRNAseq data from 105 controls")


def test_numbers_count_even_with_little_word_overlap():
    # An Arabic answer shares no English words with the passage, only the numbers.
    assert "105 controls" in best_sentence(PASSAGE, "شملت الدراسة 105 من الأصحاء [S1]")


def test_no_match_returns_none():
    assert best_sentence(PASSAGE, "Completely unrelated gut microbiome rhythm") is None
    assert best_sentence(PASSAGE, "") is None
