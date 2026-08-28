export const NBA_TEAMS_INFO = {
  "Hawks": { city: "Atlanta", primary: "#C8102E", secondary: "#FDB927", abbr: "ATL" },
  "Celtics": { city: "Boston", primary: "#007A33", secondary: "#BA9653", abbr: "BOS" },
  "Nets": { city: "Brooklyn", primary: "#000000", secondary: "#FFFFFF", abbr: "BKN" },
  "Hornets": { city: "Charlotte", primary: "#1D1160", secondary: "#00788C", abbr: "CHA" },
  "Bulls": { city: "Chicago", primary: "#CE1141", secondary: "#000000", abbr: "CHI" },
  "Cavaliers": { city: "Cleveland", primary: "#860038", secondary: "#041E42", abbr: "CLE" },
  "Mavericks": { city: "Dallas", primary: "#00538C", secondary: "#002B5E", abbr: "DAL" },
  "Nuggets": { city: "Denver", primary: "#0E2240", secondary: "#FEC524", abbr: "DEN" },
  "Pistons": { city: "Detroit", primary: "#C8102E", secondary: "#1D42BA", abbr: "DET" },
  "Warriors": { city: "Golden State", primary: "#1D428A", secondary: "#FFC72C", abbr: "GSW" },
  "Rockets": { city: "Houston", primary: "#CE1141", secondary: "#000000", abbr: "HOU" },
  "Pacers": { city: "Indiana", primary: "#002D62", secondary: "#FDBB30", abbr: "IND" },
  "Clippers": { city: "LA", primary: "#C8102E", secondary: "#1D428A", abbr: "LAC" },
  "Lakers": { city: "Los Angeles", primary: "#552583", secondary: "#FDB927", abbr: "LAL" },
  "Grizzlies": { city: "Memphis", primary: "#5D76A9", secondary: "#12173F", abbr: "MEM" },
  "Heat": { city: "Miami", primary: "#98002E", secondary: "#F9A01B", abbr: "MIA" },
  "Bucks": { city: "Milwaukee", primary: "#00471B", secondary: "#EEE1C6", abbr: "MIL" },
  "Timberwolves": { city: "Minnesota", primary: "#0C2340", secondary: "#236192", abbr: "MIN" },
  "Pelicans": { city: "New Orleans", primary: "#0C2340", secondary: "#C8102E", abbr: "NOP" },
  "Knicks": { city: "New York", primary: "#006BB6", secondary: "#F58426", abbr: "NYK" },
  "Thunder": { city: "Oklahoma City", primary: "#007AC1", secondary: "#EF3B24", abbr: "OKC" },
  "Magic": { city: "Orlando", primary: "#0077C0", secondary: "#C4CED4", abbr: "ORL" },
  "76ers": { city: "Philadelphia", primary: "#006BB6", secondary: "#ED174C", abbr: "PHI" },
  "Suns": { city: "Phoenix", primary: "#1D1160", secondary: "#E56020", abbr: "PHX" },
  "Trail Blazers": { city: "Portland", primary: "#E03A3E", secondary: "#000000", abbr: "POR" },
  "Kings": { city: "Sacramento", primary: "#5A2D81", secondary: "#63727A", abbr: "SAC" },
  "Spurs": { city: "San Antonio", primary: "#C4CED4", secondary: "#000000", abbr: "SAS" },
  "Raptors": { city: "Toronto", primary: "#CE1141", secondary: "#000000", abbr: "TOR" },
  "Jazz": { city: "Utah", primary: "#002B5C", secondary: "#F9A01B", abbr: "UTA" },
  "Wizards": { city: "Washington", primary: "#002B5C", secondary: "#E31837", abbr: "WAS" }
};

export const cmToFeetInches = (cm) => {
  const totalInches = cm / 2.54;
  const feet = Math.floor(totalInches / 12);
  const inches = Math.round(totalInches % 12);
  if (inches === 12) {
    return { feet: feet + 1, inches: 0, str: `${feet + 1}'0"` };
  }
  return { feet, inches, str: `${feet}'${inches}"` };
};

export const feetInchesToCm = (feet, inches) => {
  return Math.round((feet * 12 + inches) * 2.54 * 10) / 10;
};

export const kgToLbs = (kg) => {
  return Math.round(kg * 2.20462);
};

export const lbsToKg = (lbs) => {
  return Math.round(lbs * 0.453592 * 10) / 10;
};
