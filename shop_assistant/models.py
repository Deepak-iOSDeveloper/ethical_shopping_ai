from django.db import models
from django.contrib.auth.models import User


class SearchLog(models.Model):
    """Log user searches for analytics."""
    category    = models.CharField(max_length=100, blank=True, null=True)
    keyword     = models.CharField(max_length=200, blank=True, null=True)
    budget      = models.FloatField(null=True, blank=True)
    results_count = models.IntegerField(default=0)
    timestamp   = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Search: {self.keyword or self.category} @ {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']


class FavoriteProduct(models.Model):
    """Track favorited products (session-based)."""
    product_name = models.CharField(max_length=200)
    brand        = models.CharField(max_length=100)
    session_key  = models.CharField(max_length=40)
    added_at     = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.product_name} by {self.brand}"


class UserSavedProduct(models.Model):
    """Products scanned/entered by users — linked to Django User account."""
    user          = models.ForeignKey(User, on_delete=models.CASCADE, related_name='saved_products')
    name          = models.CharField(max_length=200)
    brand         = models.CharField(max_length=100, blank=True)
    category      = models.CharField(max_length=100, default='General')
    description   = models.TextField(blank=True)
    materials     = models.CharField(max_length=500, blank=True)
    cert          = models.CharField(max_length=300, blank=True)
    price         = models.FloatField(null=True, blank=True)
    barcode       = models.CharField(max_length=50, blank=True)

    # EcoMindNet predicted scores
    eco_score     = models.FloatField(null=True, blank=True)
    ethics_score  = models.FloatField(null=True, blank=True)
    carbon_level  = models.CharField(max_length=20, blank=True)
    overall_score = models.FloatField(null=True, blank=True)
    tags          = models.CharField(max_length=200, blank=True)

    source        = models.CharField(max_length=50, default='manual')
    saved_at      = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} — {self.user.email}"

    class Meta:
        ordering = ['-saved_at']
