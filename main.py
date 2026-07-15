import torch


def main():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # print(text)

    # getting all the unique characters from the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("".join(chars))
    print(vocab_size)

    ## Creating a mapping from characters to integers and vice versa
    string_to_integer = {ch: i for i, ch in enumerate(chars)}
    integer_to_string = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [
        string_to_integer[character] for character in s
    ]  # encode: take a string as input and output a list of integer
    decode = lambda l: "".join(
        [integer_to_string[integer] for integer in l]
    )  # decode : take a list of integer as input and output a string

    # print(encode("shishu"))
    #    print(decode(encode("shishu")))

    ## Converting the data into tensors
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data)

    ## Now spliting the data into two parts -> 1. Training data (around 90% of the dataset),
    #  2. Validation data (around 10% of the dataset)
    n = int(0.9 * len(data))
    training_data = data[:n]
    validation_data = data[n:]

    # Chunking the datasets to fed into the transformer
    block_size = 8
    training_data[: block_size + 1]


if __name__ == "__main__":
    main()
