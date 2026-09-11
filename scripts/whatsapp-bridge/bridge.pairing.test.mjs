// Pairing-by-code helpers (Moe slice E3). Pure — bridge.js is not imported,
// for the reason every test beside this one gives.
import { strict as assert } from 'node:assert';
import { normalizePairingPhone, pairingCodeDecision } from './bridge_helpers.js';

{
  // The number goes to Baileys as digits with the country code and nothing else.
  assert.equal(normalizePairingPhone('+44 7700 900000'), '447700900000');
  // A national number with no country code is still 10 digits; length
  // cannot tell, and the country code is the caller's to supply.
  assert.equal(normalizePairingPhone('(415) 555-0100'), '4155550100');
  console.log('  ✓ normalizePairingPhone strips + spaces brackets dashes');
}
{
  assert.equal(normalizePairingPhone('+1 415 555 0100'), '14155550100');
  assert.equal(normalizePairingPhone(''), null);
  assert.equal(normalizePairingPhone(undefined), null);
  assert.equal(normalizePairingPhone('12345'), null, 'five digits is not a number');
  assert.equal(normalizePairingPhone('1234567890123456'), null, 'sixteen digits is not a number');
  assert.equal(normalizePairingPhone('abc'), null);
  console.log('  ✓ normalizePairingPhone refuses what cannot be a number');
}
{
  // No phone: nothing changes for the QR path.
  assert.equal(pairingCodeDecision({ phone: null, registered: false, requested: false, qr: 'x' }), 'qr');
  // A registered socket is already a device; never ask.
  assert.equal(pairingCodeDecision({ phone: '447700900000', registered: true, requested: false, qr: 'x' }), 'none');
  // Ask exactly once, on the first update that carries a QR.
  assert.equal(pairingCodeDecision({ phone: '447700900000', registered: false, requested: false, qr: undefined }), 'wait');
  assert.equal(pairingCodeDecision({ phone: '447700900000', registered: false, requested: false, qr: 'x' }), 'request');
  assert.equal(pairingCodeDecision({ phone: '447700900000', registered: false, requested: true, qr: 'x' }), 'none');
  console.log('  ✓ pairingCodeDecision asks once, on the QR, only when unregistered');
}
console.log('\n✅ All WhatsApp pairing-code helper tests passed.');
