/**
 * Unit tests for the linked-device name.
 *
 * The phone lists a linked device under the first element of the Baileys
 * `browser` triple. That was the literal 'Hermes Agent', so every product
 * built on this bridge showed up on the person's phone under the engine's
 * name. WHATSAPP_DEVICE_NAME overrides it; absent or blank keeps the default
 * so existing deployments are unchanged.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';

import {
  DEFAULT_DEVICE_NAME,
  MAX_DEVICE_NAME_LENGTH,
  browserDescription,
  resolveDeviceName,
} from './bridge_helpers.js';

// -- default is unchanged -------------------------------------------------
{
  assert.equal(DEFAULT_DEVICE_NAME, 'Hermes Agent');
  assert.equal(resolveDeviceName({}), 'Hermes Agent');
  assert.equal(resolveDeviceName({ WHATSAPP_DEVICE_NAME: '' }), 'Hermes Agent');
  assert.equal(resolveDeviceName({ WHATSAPP_DEVICE_NAME: '   ' }), 'Hermes Agent');
  assert.equal(resolveDeviceName(undefined), 'Hermes Agent');
  console.log('  ✓ unset, empty and blank WHATSAPP_DEVICE_NAME keep the default name');
}

// -- the override lands in the browser triple's first slot ----------------
{
  assert.equal(resolveDeviceName({ WHATSAPP_DEVICE_NAME: 'Moe' }), 'Moe');
  assert.equal(resolveDeviceName({ WHATSAPP_DEVICE_NAME: '  Moe  ' }), 'Moe');
  assert.deepEqual(browserDescription({ WHATSAPP_DEVICE_NAME: 'Moe' }), ['Moe', 'Chrome', '120.0']);
  assert.deepEqual(browserDescription({}), ['Hermes Agent', 'Chrome', '120.0']);
  console.log('  ✓ WHATSAPP_DEVICE_NAME replaces only the name; browser and version are untouched');
}

// -- a runaway value is cut to what a phone can show ----------------------
{
  const long = 'x'.repeat(MAX_DEVICE_NAME_LENGTH + 40);
  assert.equal(resolveDeviceName({ WHATSAPP_DEVICE_NAME: long }).length, MAX_DEVICE_NAME_LENGTH);
  console.log(`  ✓ the name is capped at ${MAX_DEVICE_NAME_LENGTH} characters`);
}

// -- bridge.js actually uses it: the literal must be gone from the socket --
{
  const src = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');
  assert.equal(src.includes("browser: ['Hermes Agent'"), false,
    'bridge.js still hardcodes the device name in makeWASocket');
  assert.equal(src.includes('browser: browserDescription()'), true,
    'bridge.js does not pass browserDescription() to makeWASocket');
  console.log('  ✓ makeWASocket takes its browser triple from browserDescription()');
}

console.log('bridge.devicename.test.mjs: all passed');
