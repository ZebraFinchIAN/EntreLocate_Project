import fs from "fs/promises";
import fetch from "node-fetch";

async function fetchAllDistrictsGroupedByCity() {
  const query = `
    [out:json][timeout:180];
    area["ISO3166-1"="TR"][admin_level=2]->.searchArea;

    // Provinces (admin_level=4)
    rel(area.searchArea)["admin_level"="4"];
    out ids tags;

    // Districts (admin_level=6)
    rel(area.searchArea)["admin_level"="6"];
    out ids tags;
  `;

  const response = await fetch("https://overpass-api.de/api/interpreter", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `data=${encodeURIComponent(query)}`,
  });

  if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`);
  const data = await response.json();

  const provinces = {};
  const districts = [];

  for (const el of data.elements) {
    if (el.tags?.["admin_level"] === "4") {
      provinces[el.id] = el.tags.name;
    } else if (el.tags?.["admin_level"] === "6") {
      districts.push(el);
    }
  }

  console.log(`Provinces: ${Object.keys(provinces).length}`);
  console.log(`Districts: ${districts.length}`);

  const results = {};

  for (const district of districts) {
    const districtName = district.tags.name;
    const parentProvinceId = district.tags["is_in:province"] || district.tags["addr:province"];

    if (parentProvinceId) {
      // This usually fails because tags don’t include these values
      continue;
    }

    // Try to match based on known relations — fallback method
    const matchedProvince = Object.entries(provinces).find(([id, name]) => {
      return district.tags.name?.includes(name); // fuzzy match (optional)
    });

    if (matchedProvince) {
      const [, provinceName] = matchedProvince;
      if (!results[provinceName]) results[provinceName] = [];
      results[provinceName].push(districtName);
    }
  }

  const sorted = Object.keys(results)
    .sort()
    .reduce((acc, city) => {
      acc[city] = results[city].sort();
      return acc;
    }, {});

  await fs.writeFile("cities_with_districts.json", JSON.stringify(sorted, null, 2));
  console.log(`Saved ${Object.keys(sorted).length} cities with districts to cities_with_districts.json`);
}

fetchAllDistrictsGroupedByCity().catch(console.error);
