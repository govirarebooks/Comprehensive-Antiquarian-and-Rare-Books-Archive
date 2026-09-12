import sys

from sentence_transformers import CrossEncoder


MODEL_NAME = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python rag/test_cross_encoder.py "query"'
        )

    query = " ".join(sys.argv[1:])

    model = CrossEncoder(
        MODEL_NAME
    )

    passages = [
        (
            "A sixteenth-century book concerning "
            "oracles, prophecy, witchcraft, astrology "
            "and celestial phenomena."
        ),
        (
            "A sixteenth-century collection of letters "
            "by a German humanist concerning philology, "
            "correspondence and literary scholarship."
        ),
        (
            "A treatise on palmistry and divination, "
            "describing the interpretation of lines and "
            "symbols on the human hand."
        ),
    ]

    pairs = [
        [query, passage]
        for passage in passages
    ]

    scores = model.predict(
        pairs
    )

    ranked = sorted(
        zip(scores, passages),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    print()
    print(
        "GOVI CROSS-ENCODER TEST"
    )
    print("=" * 80)

    for index, (
        score,
        passage,
    ) in enumerate(
        ranked,
        start=1,
    ):
        print()
        print(
            f"{index}. "
            f"Score: {float(score):.4f}"
        )
        print(
            passage
        )


if __name__ == "__main__":
    main()
