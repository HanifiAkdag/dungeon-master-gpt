def radio_to_fandom_name(radio_input=None):
    table = {
        "Harry Potter": "harrypotter",
        "Star Wars": "starwars",
        "Lord of the Rings": "lotr",
        "Marvel": "marvel",
        "DC": "dc"
    }
    if radio_input in table.values():
        return radio_input
    
    return table.get(radio_input, None)