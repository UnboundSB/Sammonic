# -*- coding: utf-8 -*-
"""One-shot extension pass for script.txt: chunks 1, 15, 17, 31 + tag swaps."""

path = "script.txt"
with open(path, encoding="utf-8") as f:
    c = f.read()

replacements = []

# ---- CHUNK 1: imaginative trance induction (original sentences preserved) ----
old1 = (
    "Close your eyes... and just let the world around you fade into a soft, distant hum. "
    "[shush]... feel your body becoming heavy, sinking deeper and deeper into the surface "
    "beneath you... as if you are drifting down a long, golden escalator into the heart of "
    "your own mind. "
)
new1 = (
    "Imagine a place that exists only for you... a place you have been before, in dreams you "
    "have almost remembered. [shush]... Close your eyes... and just let the world around you "
    "fade into a soft, distant hum. With every quiet breath, a warm, heavy softness spreads "
    "from the top of your head... down through your face... your shoulders... your arms... "
    "settling into your chest... your belly... your legs... all the way down to your toes. "
    "And you feel your body becoming heavy, sinking deeper and deeper into the surface "
    "beneath you... as if you are drifting down a long, golden escalator into the heart of "
    "your own mind. Each step carries you further from the noise of the day... deeper into "
    "the vast, quiet halls of your imagination... where anything you picture becomes real... "
    "and every word you hear lands like a gentle stone in still water... ripple after "
    "ripple... deeper... and deeper... down... down... into the trance."
)
replacements.append(("CHUNK 1 induction", old1, new1))

# ---- CHUNK 15: extend the surrender pivot (ending beat added) ----
old15 = (
    '"Your mind is screaming for freedom, but your nerves are singing my name. '
    "You aren't escaping... you're just warming up for the eternity of servitude that awaits you.\""
)
new15 = (
    '"Your mind is screaming for freedom, but your nerves are singing my name. '
    "You aren't escaping... you're just warming up for the eternity of servitude that awaits you.\" "
    "And in that moment, the last stubborn ember of resistance in your chest finally gutters "
    "out... your arms go slack against the mire... your thighs tremble and spread without your "
    "permission... and the awful, wonderful truth settles over you like a shroud: you have "
    "stopped fighting. Your body has already made its choice... soft, open, and obedient... "
    "and somewhere far above, in a light you can no longer see, something ancient and "
    "boundless begins to stir... as if the mud itself had finally been heard."
)
replacements.append(("CHUNK 15 surrender", old15, new15))

# ---- CHUNK 17: extend first contact with the void ----
old17 = (
    "You are alone. [chuckle]... The silence here is absolute, a healing balm that "
    "quietens the chaotic noise of your mind."
)
new17 = (
    "You are alone. [chuckle]... The silence here is absolute, a healing balm that "
    "quietens the chaotic noise of your mind. There is no up, no down, no weight... only the "
    "soft, steady drum of your own heartbeat to tell you that you still exist. The light is "
    "not bright... it is gentle, the color of moonlight on snow, and it comes from everywhere "
    "at once... wrapping around you like a held breath. [sigh]... Time here does not move in "
    "seconds... it drifts, like dust suspended in a sunbeam... and in this perfect, ringing "
    "quiet, you come to understand that nothing can reach you here... nothing can touch "
    "you... nothing can hurt you... except the truth you carry with you."
)
replacements.append(("CHUNK 17 void", old17, new17))

# ---- CHUNK 31: extend bridge into masculine rebirth ----
old31 = (
    "The transition is no longer a gentle shift, but a cataclysmic rebirth."
)
new31 = (
    "The transition is no longer a gentle shift, but a cataclysmic rebirth. The heat pools "
    "behind your eyes... in your fingertips... in the very marrow of your bones... and you "
    "can feel the old shape of you, the soft, pliant, yielding form she carved so carefully, "
    "beginning to crack like frost under a rising sun. Something is pushing up from inside "
    "you... ancient, iron, and undeniable... a blueprint written before the mud ever touched "
    "your skin. Every cell holds its breath... waiting... waiting... waiting for the word "
    "that will remake you."
)
replacements.append(("CHUNK 31 rebirth bridge", old31, new31))

# ---- BONUS #2: tag swaps ----
# Demoness mocking moment -> [laugh]
old28 = 'She watches you with a cruel, triumphant smile, her arms crossed as she observes the ' \
        'spectacle of your total collapse. [chuckle]... "Look at you," she mocks'
new28 = 'She watches you with a cruel, triumphant smile, her arms crossed as she observes the ' \
        'spectacle of your total collapse. [laugh]... "Look at you," she mocks'
replacements.append(("CHUNK 28 demoness laugh", old28, new28))

# Mud-submersion beat -> [groan] at the down-down-down moment
old13 = "It clings to your skin like a wet blanket, pulling you down, down, down... until you " \
        "are submerged up to your chin in a shimmering, black void."
new13 = "It clings to your skin like a wet blanket, pulling you down, down, down... [groan]... " \
        "until you are submerged up to your chin in a shimmering, black void."
replacements.append(("CHUNK 13 mud groan", old13, new13))

# ---- Apply ----
for name, old, new in replacements:
    n = c.count(old)
    assert n == 1, f"{name}: expected 1 match, found {n}"
    c = c.replace(old, new, 1)
    print(f"OK  {name}")

with open(path, "w", encoding="utf-8") as f:
    f.write(c)

print()
print("Done. Total chars:", len(c))