from tts.yuuna_reply import parse_gemma_output, build_style_prompt, format_display_text

r = parse_gemma_output('{"text":"Hi!","emotion":"cheerful","intensity":0.9}')
print("JSON:", r)
print("Style:", build_style_prompt(r))
print("Display:", format_display_text(r))

r2 = parse_gemma_output("[HAPPY] plain fallback")
print("Fallback:", r2)
