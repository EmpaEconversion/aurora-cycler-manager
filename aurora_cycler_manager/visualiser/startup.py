"""Copyright © 2025, Empa.

Minimal script to show a logo and start logging immediately on starting the app.
Initialization can take a few seconds if connecting to several cyclers.
Show stuff happening in the terminal to keep users happy :).
"""

import logging

from aurora_cycler_manager.setup_logging import setup_logging

logger = logging.getLogger(__name__)

ascii_art = r"""
       AURO                                                                     
      RAUROR                                                                    
     AURORAUR                               OR      AU                          
     RORAUROR                              AURO    RAUR                         
    ORAU  RORA     UROR     AURO    RAURO   RAURORAURO      RAUROR   AURORAURO  
    RAUR  ORAU     RORA     UROR   AURORA   UROR  AURO     RAURORA  URORAURORAU 
   RORA    UROR    AURO     RAUR  ORAU     ROR      AUR   ORAU     RORAU   RORAU
  RORAU    RORAU   RORA     UROR  AURO   RAURO      RAURO RAUR     ORAU     RORA
 URORAURORAURORAU  RORA     UROR  AURO   RAURO      RAURO RAUR     ORAU     RORA
 URORAURORAURORAU  RORAU   RORAU  RORA     URO      RAU   RORA     URORA   URORA
URORA         UROR  AURORAURORA   UROR      AU      RO    RAUR      ORAURORAUROR
AUROR         AURO   RAURORAUR    ORAU      RORAURORAU    RORA       URORAUR ORA
                                           UROR    AURO                         
                                            RA      UR                          
"""  # noqa: W291


def rgb(r: int, g: int, b: int, text: str) -> str:
    """Format text with RGB color."""
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def gradient_line(start_rgb: tuple, mid_rgb: tuple, end_rgb: tuple, text: str) -> str:
    """Create a gradient effect for a line of text."""
    length = len(text)
    gradient = []
    for i in range(length):
        if i < length / 2:
            r = int(start_rgb[0] + (mid_rgb[0] - start_rgb[0]) * i / (length / 2 - 1))
            g = int(start_rgb[1] + (mid_rgb[1] - start_rgb[1]) * i / (length / 2 - 1))
            b = int(start_rgb[2] + (mid_rgb[2] - start_rgb[2]) * i / (length / 2 - 1))
        else:
            r = int(mid_rgb[0] + (end_rgb[0] - mid_rgb[0]) * (i - length / 2) / (length / 2 - 1))
            g = int(mid_rgb[1] + (end_rgb[1] - mid_rgb[1]) * (i - length / 2) / (length / 2 - 1))
            b = int(mid_rgb[2] + (end_rgb[2] - mid_rgb[2]) * (i - length / 2) / (length / 2 - 1))
        gradient.append(rgb(r, g, b, text[i]))
    return "".join(gradient)


def main() -> None:
    """Start the Aurora app."""
    for line in ascii_art.splitlines():
        gradient_text = gradient_line((103, 203, 243), (77, 193, 185), (183, 119, 179), line)
        print(gradient_text)
    print()

    setup_logging()
    logger.info("Trying to connect to cycler servers...")
    from aurora_cycler_manager.visualiser.app import main as app_main  # noqa: PLC0415

    logger.info("Starting Aurora app...")
    app_main()


if __name__ == "__main__":
    main()
