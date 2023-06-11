def can_convert_to_int(text: str) -> bool:
    try:
        int(text)

        return True
    except ValueError:
        return False
