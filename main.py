import torch


def main():
    # PyTorch config to run the pytorch into cpu
    #  
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Metal Performance Shaders (MPS)
    
    print(f"Using device: {device}")

    # Main code 
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
    # print("len data", len(data))

    ## Now spliting the data into two parts -> 1. Training data (around 90% of the dataset),
    #  2. Validation data (around 10% of the dataset)
    n = int(0.9 * len(data))
    training_data = data[:n]
    validation_data = data[n:]

    # Chunking the datasets to feed into the transformer
    # Works by feeding last n token (max upto 8, since here block_size is = 8) to predict the next token 
    # These are for one by one sequencial processing of the data's in the gpu 
    block_size = 8
    training_data[: block_size + 1]

    x = training_data[:block_size]
    y = training_data[1:block_size+1]
    # print("range of blocksize", range(block_size))
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context} the target is: {target} ")


    # Introducing Batching for efficient processing on gpu efficiently 
    
    torch.manual_seed(1337)
    batch_size = 4 #how many independent sequence will be processed in parallel
    # block_size = 8 (already defined above)
    def get_batch(split):
    # generate a small batch of data for input x and target y 
        data = training_data if split == "train" else validation_data
        inputx = torch.randint(len(data)-block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in inputx])
        y = torch.stack([data[i+1: i+block_size+1] for i in inputx])
        return x, y

    xb, yb = get_batch("train")
    print("inputs:")
    print(xb.shape)
    print("targets:")
    print(yb.shape)
    print(yb)

    print('------------------------------')
    
    for b in range(batch_size): # batch Dimension
        for t in range(block_size): # time dimension 
         context = xb[b, :t+1]
         target = yb[b, t]
         print(f"when input is {context.tolist()} the target is: {target} ")
    
    


if __name__ == "__main__":
    main()
